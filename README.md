# Push-Optimal Sokoban Solver

A Sokoban solver written in plain JavaScript (Node 22, standard
library only) for the make.fun competition "Build the best Sokoban
solver, 21 November 2026".

## How it works

Every level is a hallway with one to four small rooms off it, and
every box stays in its room, so the solver:

1. Parses the XSB level and computes the live squares (squares from
   which a box can still be pushed onto a goal) by pulling backwards
   from the goals.
2. Splits the level into rooms by dropping the hallway row from the
   floor graph and taking the connected components that hold boxes.
3. Solves each room push-optimally with A* over pushes, using a
   minimum-assignment heuristic over walking distances, dead-square
   pruning, and frozen 2x2 deadlock pruning, all under a fixed
   state budget.
4. Stitches the room solutions together with hallway walks, and
   replays the full move string to verify it before writing it out.

When the split fails, a whole-level push search is the fallback; when
nothing verifies, the solver writes an empty (valid, zero-point)
answer rather than an illegal move string.

## Running it

```sh
node main.ts
```

It reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt`. For local runs, set `MAKEFUN_INPUT_DIR`
and `MAKEFUN_OUTPUT_DIR`. The run is deterministic: fixed iteration
orders and fixed state budgets, no clock reads, no randomness.

On the practice set it scores 1,000,000: every instance solved in
exactly its push-optimal count (13, 16, 29, 47, 60 pushes).
