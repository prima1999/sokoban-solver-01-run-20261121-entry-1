// Push-optimal Sokoban solver.
//
// How it works: cut each level into its rooms, solve every room
// with a push-optimal A* search, then join the room solutions with
// walks along the hallway. When the cut fails, a whole-level search
// is the fallback. The program is deterministic: fixed iteration
// orders, a fixed state budget per search, no clock reads and no
// randomness anywhere.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';

// The four steps, in one fixed order shared by every search and walk.
const DIRS = [
  { dx: 0, dy: -1, ch: "u" },
  { dx: 0, dy: 1, ch: "d" },
  { dx: -1, dy: 0, ch: "l" },
  { dx: 1, dy: 0, ch: "r" },
];

// Work budgets, counted in states rather than wall-clock time. A
// room search that exhausts its budget fails that one instance
// instead of eating the run's whole time budget; the same holds for
// the whole-level fallback, which is only a last resort.
const STATE_CAP_WHOLE = 5000;
const STATE_CAP_ROOM = 100000;

// A parsed level: walls, goals, starting boxes and player, on a
// grid of width w and height h indexed y * w + x. Short rows pad
// out as walls, the same way the checker reads them.
function parseLevel(text) {
  const lines = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...lines.map((r) => r.length));
  const h = lines.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes = [];
  let player = -1;
  lines.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goal, boxes, player };
}

// Live squares: squares from which a box can still be pushed onto
// some goal. Found by working backwards from the goals: a square t
// is live when a box on t can be pushed one step onto a live
// square, which needs the square behind t (opposite the push) to be
// open floor. Whether the player can actually reach that standing
// square is ignored, so this set safely over-approximates: a box
// off a live square is certainly stuck, and pruning it can never
// throw away a real solution.
function computeLive(L) {
  const live = new Uint8Array(L.w * L.h);
  const work = [];
  for (let i = 0; i < L.w * L.h; i++) {
    if (L.goal[i]) {
      live[i] = 1;
      work.push(i);
    }
  }
  for (let qi = 0; qi < work.length; qi++) {
    const s = work[qi];
    const sx = s % L.w;
    const sy = (s / L.w) | 0;
    for (const d of DIRS) {
      const tx = sx - d.dx;
      const ty = sy - d.dy;
      if (tx < 0 || ty < 0 || tx >= L.w || ty >= L.h) continue;
      const t = ty * L.w + tx;
      if (L.wall[t] || live[t]) continue;
      const bx = sx - 2 * d.dx;
      const by = sy - 2 * d.dy;
      if (bx < 0 || by < 0 || bx >= L.w || by >= L.h) continue;
      if (L.wall[by * L.w + bx]) continue;
      live[t] = 1;
      work.push(t);
    }
  }
  return live;
}

// The squares the player can reach from a starting square without
// moving any box: a flood fill over squares that are not solid.
// Returns the reached set plus its lowest square, which names the
// region (two states with the same boxes and the same lowest
// reachable square share one reachable region).
function regionOf(L, start, solid) {
  const seen = new Uint8Array(L.w * L.h);
  const stack = [start];
  seen[start] = 1;
  let min = start;
  while (stack.length) {
    const cur = stack.pop();
    if (cur < min) min = cur;
    const cx = cur % L.w;
    const cy = (cur / L.w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
      const ni = ny * L.w + nx;
      if (seen[ni] || solid[ni]) continue;
      seen[ni] = 1;
      stack.push(ni);
    }
  }
  return { seen, min };
}

// Shortest walk from one square to any square of a target set,
// with solid squares treated as walls. Returns the move string, or
// null when no target can be reached. Neighbours are tried in DIRS
// order, so the walk comes out the same on every run.
function walkPath(L, from, targets, solid) {
  if (targets.has(from)) return '';
  const prev = new Int32Array(L.w * L.h).fill(-1);
  const stepCh = new Array(L.w * L.h).fill('');
  const queue = [from];
  prev[from] = from;
  let hit = -1;
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi];
    if (targets.has(cur)) {
      hit = cur;
      break;
    }
    const cx = cur % L.w;
    const cy = (cur / L.w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
      const ni = ny * L.w + nx;
      if (prev[ni] !== -1 || solid[ni]) continue;
      prev[ni] = cur;
      stepCh[ni] = d.ch;
      queue.push(ni);
    }
  }
  if (hit === -1) return null;
  let out = '';
  let cur = hit;
  while (cur !== from) {
    out = stepCh[cur] + out;
    cur = prev[cur];
  }
  return out;
}

// Walking distances from one square over open floor, ignoring the
// boxes. Used by the push heuristic: a box must be pushed at least
// this far to reach a goal, whatever the other boxes do.
function bfsDist(L, from) {
  const dist = new Int32Array(L.w * L.h).fill(-1);
  dist[from] = 0;
  const queue = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi];
    const cx = cur % L.w;
    const cy = (cur / L.w) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
      const ni = ny * L.w + nx;
      if (dist[ni] !== -1 || L.wall[ni]) continue;
      dist[ni] = dist[cur] + 1;
      queue.push(ni);
    }
  }
  return dist;
}

// Cheapest way to match every box to a distinct goal, by walking
// distance. With few boxes this is the exact assignment (every
// permutation is tried); with many boxes it is the sum of each box's
// nearest goal, which is still a lower bound on the pushes left.
// Either way the value never overestimates, so A* stays
// push-optimal.
function heuristic(boxes, goals, distFromGoal) {
  const n = boxes.length;
  const costs = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < goals.length; j++) {
      row.push(distFromGoal[j][boxes[i]]);
    }
    costs.push(row);
  }
  if (n <= 6) {
    let best = Infinity;
    const taken = new Array(n).fill(false);
    const rec = (i, sum) => {
      if (sum >= best) return;
      if (i === n) {
        best = sum;
        return;
      }
      for (let j = 0; j < n; j++) {
        if (taken[j]) continue;
        const c = costs[i][j];
        if (c < 0) continue;
        taken[j] = true;
        rec(i + 1, sum + c);
        taken[j] = false;
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let m = Infinity;
    for (let j = 0; j < goals.length; j++) {
      const c = costs[i][j];
      if (c >= 0 && c < m) m = c;
    }
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

// A box layout that can never be solved: a box off its goal on a
// non-live square, or a frozen 2x2 block of walls and boxes holding
// a box off its goal (no box inside such a block can ever move).
function isDead(boxes, L, live) {
  const set = new Set(boxes);
  for (const b of boxes) {
    if (!L.goal[b] && !live[b]) return true;
  }
  for (let y = 0; y + 1 < L.h; y++) {
    for (let x = 0; x + 1 < L.w; x++) {
      const cells = [y * L.w + x, y * L.w + x + 1, (y + 1) * L.w + x, (y + 1) * L.w + x + 1];
      let allSolid = true;
      let bad = false;
      for (const c of cells) {
        if (L.wall[c]) continue;
        if (set.has(c)) {
          if (!L.goal[c]) bad = true;
        } else {
          allSolid = false;
          break;
        }
      }
      if (allSolid && bad) return true;
    }
  }
  return false;
}

// Whether the box just pushed onto dest is stuck: off its goal on
// a non-live square, or inside a frozen 2x2 block of walls and
// boxes. Only the moved box can newly deadlock, because the state
// it came from had already passed this check, so this is the fast
// per-successor version of the check above.
function isDeadAfter(dest, boxSet, L, live) {
  if (!L.goal[dest] && !live[dest]) return true;
  const dx = dest % L.w;
  const dy = (dest / L.w) | 0;
  for (let oy = -1; oy <= 0; oy++) {
    for (let ox = -1; ox <= 0; ox++) {
      const x = dx + ox;
      const y = dy + oy;
      if (x < 0 || y < 0 || x + 1 >= L.w || y + 1 >= L.h) continue;
      let allSolid = true;
      let bad = false;
      for (let j = 0; j < 4; j++) {
        const c = (y + (j >> 1)) * L.w + x + (j & 1);
        if (L.wall[c]) continue;
        if (boxSet.has(c)) {
          if (!L.goal[c]) bad = true;
        } else {
          allSolid = false;
          break;
        }
      }
      if (allSolid && bad) return true;
    }
  }
  return false;
}

// A small binary heap ordered by (f, h, insertion order), so the
// order states leave the search is fully deterministic.
function makeHeap() {
  const items = [];
  const order = (s, t) => {
    if (s.f !== t.f) return s.f - t.f;
    if (s.h !== t.h) return s.h - t.h;
    return s.seq - t.seq;
  };
  const swap = (i, j) => {
    const tmp = items[i];
    items[i] = items[j];
    items[j] = tmp;
  };
  return {
    size: () => items.length,
    push: (s) => {
      items.push(s);
      let i = items.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (order(items[p], items[i]) <= 0) break;
        swap(p, i);
        i = p;
      }
    },
    pop: () => {
      const top = items[0];
      const last = items.pop();
      if (items.length > 0) {
        items[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1;
          const r = l + 1;
          let m = i;
          if (l < items.length && order(items[l], items[m]) < 0) m = l;
          if (r < items.length && order(items[r], items[m]) < 0) m = r;
          if (m === i) break;
          swap(m, i);
          i = m;
        }
      }
      return top;
    },
  };
}

// Solid squares for a state: a fixed base (walls plus fixed boxes)
// with the moving boxes marked on top. Copying the base is one
// fast block copy instead of a scan of the whole board.
function solidFor(baseSolid, boxes) {
  const solid = new Uint8Array(baseSolid);
  for (const b of boxes) solid[b] = 1;
  return solid;
}

// Push-optimal A* over pushes. A state is the sorted moving-box
// squares plus the player's reachable region (named by its lowest
// square). The first solved state taken off the heap is
// push-optimal, because the heuristic is admissible and consistent.
// Returns the move string, or null when the state budget runs out.
function pushSearch(L, live, startBoxes, goals, startPlayer, fixed, roomSet, cap) {
  const goalSet = new Set(goals);
  const distFromGoal = goals.map((g) => bfsDist(L, g));
  const keyOf = (boxes, regionMin) => boxes.join(',') + '|' + regionMin;
  const states = new Map();
  const best = new Map();
  const heap = makeHeap();
  let seq = 0;
  const baseSolid = new Uint8Array(L.w * L.h);
  for (let i = 0; i < L.w * L.h; i++) {
    if (L.wall[i] || fixed[i]) baseSolid[i] = 1;
  }
  const startSolid = solidFor(baseSolid, startBoxes);
  const startRegion = regionOf(L, startPlayer, startSolid);
  const startH = heuristic(startBoxes, goals, distFromGoal);
  if (startH === Infinity) return null;
  if (isDead(startBoxes, L, live)) return null;
  const start = {
    boxes: [...startBoxes].sort((a, b) => a - b),
    player: startPlayer,
    g: 0,
    h: startH,
    f: startH,
    seq: seq++,
    key: '',
    parent: null,
    pushBox: -1,
    pushDir: -1,
  };
  start.key = keyOf(start.boxes, startRegion.min);
  states.set(start.key, start);
  best.set(start.key, 0);
  heap.push(start);
  let expanded = 0;
  while (heap.size() > 0) {
    const cur = heap.pop();
    if (cur.g !== best.get(cur.key)) continue;
    if (cur.boxes.every((b) => goalSet.has(b))) return rebuild(L, fixed, cur, states);
    if (expanded++ >= cap) return null;
    const solid = solidFor(baseSolid, cur.boxes);
    const region = regionOf(L, cur.player, solid);
    for (const b of cur.boxes) {
      const bx = b % L.w;
      const by = (b / L.w) | 0;
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di];
        const sx = bx - d.dx;
        const sy = by - d.dy;
        const tx = bx + d.dx;
        const ty = by + d.dy;
        if (sx < 0 || sy < 0 || sx >= L.w || sy >= L.h) continue;
        if (tx < 0 || ty < 0 || tx >= L.w || ty >= L.h) continue;
        const stand = sy * L.w + sx;
        const dest = ty * L.w + tx;
        if (!region.seen[stand]) continue;
        if (solid[dest]) continue;
        if (roomSet !== null && !roomSet.has(dest)) continue;
        const next = cur.boxes.map((v) => (v === b ? dest : v));
        next.sort((x, y) => x - y);
        if (isDeadAfter(dest, new Set(next), L, live)) continue;
        const h = heuristic(next, goals, distFromGoal);
        if (h === Infinity) continue;
        const g2 = cur.g + 1;
        const solid2 = solidFor(baseSolid, next);
        const region2 = regionOf(L, b, solid2);
        const key = keyOf(next, region2.min);
        if (best.has(key) && best.get(key) <= g2) continue;
        best.set(key, g2);
        const state = {
          boxes: next,
          player: b,
          g: g2,
          h,
          f: g2 + h,
          seq: seq++,
          key,
          parent: cur.key,
          pushBox: b,
          pushDir: di,
        };
        states.set(key, state);
        heap.push(state);
      }
    }
  }
  return null;
}

// Turn a solved search state into moves: walk the parent chain
// back, rebuilding each push's walk from the parent state's player
// square to the pushing square, then the push itself.
function rebuild(L, fixed, solved, states) {
  const chain = [];
  let cur = solved;
  while (cur.parent !== null) {
    chain.push(cur);
    cur = states.get(cur.parent);
  }
  chain.reverse();
  const first = chain.length > 0 ? states.get(chain[0].parent) : solved;
  let player = first.player;
  let out = '';
  const baseSolid = new Uint8Array(L.w * L.h);
  for (let i = 0; i < L.w * L.h; i++) {
    if (L.wall[i] || fixed[i]) baseSolid[i] = 1;
  }
  for (const step of chain) {
    const parent = states.get(step.parent);
    const d = DIRS[step.pushDir];
    const bx = step.pushBox % L.w;
    const by = (step.pushBox / L.w) | 0;
    const stand = (by - d.dy) * L.w + (bx - d.dx);
    const solid = solidFor(baseSolid, parent.boxes);
    const walk = walkPath(L, player, new Set([stand]), solid);
    if (walk === null) return null;
    out += walk + d.ch;
    player = step.pushBox;
  }
  return out;
}

// Apply a move string to a box layout, returning the new boxes and
// player square, or null when any step is illegal. Used to track the
// board while joining rooms, and to verify a full answer.
function applyMoves(L, boxes, player, moves) {
  const set = new Set(boxes);
  let p = player;
  for (const ch of moves) {
    const d = DIRS.find((v) => v.ch === ch);
    if (d === undefined) return null;
    const nx = (p % L.w) + d.dx;
    const ny = ((p / L.w) | 0) + d.dy;
    if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) return null;
    const next = ny * L.w + nx;
    if (L.wall[next]) return null;
    if (set.has(next)) {
      const bx = nx + d.dx;
      const by = ny + d.dy;
      if (bx < 0 || by < 0 || bx >= L.w || by >= L.h) return null;
      const beyond = by * L.w + bx;
      if (L.wall[beyond] || set.has(beyond)) return null;
      set.delete(next);
      set.add(beyond);
    }
    p = next;
  }
  return { boxes: [...set].sort((a, b) => a - b), player: p };
}

// Replay a move string from the level's start. Returns whether it
// is legal, whether it solves the level, and its push count, using
// the same rules as the checker.
function replayMoves(L, moves) {
  const done = applyMoves(L, L.boxes, L.player, moves);
  if (done === null) return { legal: false, solved: false, pushes: 0 };
  const solved = done.boxes.every((b) => L.goal[b]);
  return { legal: true, solved, pushes: countPushes(L, moves) };
}

// Count the steps of a move string that move a box.
function countPushes(L, moves) {
  const set = new Set(L.boxes);
  let p = L.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = DIRS.find((v) => v.ch === ch);
    if (d === undefined) return -1;
    const px = p % L.w;
    const py = (p / L.w) | 0;
    const next = (py + d.dy) * L.w + (px + d.dx);
    if (set.has(next)) {
      const beyond = next + d.dy * L.w + d.dx;
      set.delete(next);
      set.add(beyond);
      pushes++;
    }
    p = next;
  }
  return pushes;
}

// Split a level into rooms: drop the hallway row (the player's
// row) from the floor graph and take the connected components that
// hold boxes. Each room keeps its own boxes, goals and squares, plus
// its door square: the room square next to a hallway floor square.
function findRooms(L) {
  const hallRow = (L.player / L.w) | 0;
  const comp = new Int32Array(L.w * L.h).fill(-1);
  const rooms = [];
  for (let i = 0; i < L.w * L.h; i++) {
    if (L.wall[i] || ((i / L.w) | 0) === hallRow || comp[i] !== -1) continue;
    const id = rooms.length;
    const squares = new Set();
    const boxes = [];
    const goals = [];
    const queue = [i];
    comp[i] = id;
    for (let qi = 0; qi < queue.length; qi++) {
      const cur = queue[qi];
      squares.add(cur);
      if (L.goal[cur]) goals.push(cur);
      if (L.boxes.includes(cur)) boxes.push(cur);
      const cx = cur % L.w;
      const cy = (cur / L.w) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= L.w || ny >= L.h) continue;
        const ni = ny * L.w + nx;
        if (L.wall[ni] || ((ni / L.w) | 0) === hallRow || comp[ni] !== -1) continue;
        comp[ni] = id;
        queue.push(ni);
      }
    }
    rooms.push({ squares, boxes: boxes.sort((a, b) => a - b), goals: goals.sort((a, b) => a - b), entry: -1 });
  }
  const filled = rooms.filter((r) => r.boxes.length > 0);
  for (const r of filled) {
    let entry = -1;
    for (const s of [...r.squares].sort((a, b) => a - b)) {
      const sx = s % L.w;
      const sy = (s / L.w) | 0;
      for (const d of DIRS) {
        const ny = sy + d.dy;
        if (ny !== hallRow) continue;
        const nx = sx + d.dx;
        if (nx < 0 || nx >= L.w) continue;
        const ni = ny * L.w + nx;
        if (!L.wall[ni]) {
          entry = s;
          break;
        }
      }
      if (entry !== -1) break;
    }
    r.entry = entry;
  }
  return { rooms: filled, hallRow };
}

// Solve one level: split it into rooms, solve each room
// push-optimally in turn, and join the pieces with hallway walks.
// Falls back to a whole-level search, and to an empty answer when
// nothing verifies.
function solveLevel(text) {
  const L = parseLevel(text);
  const live = computeLive(L);
  const allGoals = [];
  for (let i = 0; i < L.w * L.h; i++) if (L.goal[i]) allGoals.push(i);
  const stitched = solveByRooms(L, live);
  if (stitched !== null) {
    const rep = replayMoves(L, stitched);
    if (rep.legal && rep.solved) return stitched;
  }
  const whole = pushSearch(L, live, L.boxes, allGoals, L.player, new Uint8Array(L.w * L.h), null, STATE_CAP_WHOLE);
  if (whole !== null) {
    const rep = replayMoves(L, whole);
    if (rep.legal && rep.solved) return whole;
  }
  return '';
}

// Room-by-room solve. Returns the joined move string, or null when
// the split or any room search fails.
function solveByRooms(L, live) {
  const { rooms, hallRow } = findRooms(L);
  const placed = new Set();
  for (const r of rooms) {
    if (r.boxes.length !== r.goals.length) return null;
    if (r.entry === -1) return null;
    for (const b of r.boxes) placed.add(b);
  }
  if (placed.size !== L.boxes.length) return null;
  rooms.sort((a, b) => a.entry - b.entry);
  let curPlayer = L.player;
  let curBoxes = [...L.boxes].sort((a, b) => a - b);
  let out = '';
  for (const room of rooms) {
    const ex = room.entry % L.w;
    const ey = (room.entry / L.w) | 0;
    const targets = new Set();
    for (const d of DIRS) {
      const nx = ex + d.dx;
      const ny = ey + d.dy;
      if (ny !== hallRow) continue;
      if (nx < 0 || nx >= L.w) continue;
      const ni = ny * L.w + nx;
      if (!L.wall[ni]) targets.add(ni);
    }
    if (targets.size === 0) return null;
    const solid = solidFor(L.wall, curBoxes);
    const walk = walkPath(L, curPlayer, targets, solid);
    if (walk === null) return null;
    const walked = applyMoves(L, curBoxes, curPlayer, walk);
    if (walked === null) return null;
    curPlayer = walked.player;
    curBoxes = walked.boxes;
    out += walk;
    const fixed = new Uint8Array(L.w * L.h);
    for (const b of curBoxes) {
      if (!room.boxes.includes(b)) fixed[b] = 1;
    }
    const seg = pushSearch(L, live, room.boxes, room.goals, curPlayer, fixed, room.squares, STATE_CAP_ROOM);
    if (seg === null) return null;
    const solved = applyMoves(L, curBoxes, curPlayer, seg);
    if (solved === null) return null;
    curPlayer = solved.player;
    curBoxes = solved.boxes;
    out += seg;
  }
  return out;
}

// Answer every instance in the input directory.
for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}
