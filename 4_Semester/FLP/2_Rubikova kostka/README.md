# Rubik’s Cube Solver

**Author:** Samuel Kuchta  
**Login:** xkucht11  
**Date:** April 24, 2025  
**Project:** flp24-log, Rubik’s Cube solver  


## Description of the Solution Method

The program reads a 3×3×3 Rubik’s Cube state from standard input (using helper predicates from the `input2.pl`), parses it into an internal representation, then searches for an optimal solution via **Iterative Deepening Depth-First Search (IDDFS)**.

1. **Internal Representation**  
  - Each face of the cube (Front, Right, Back, Left, Up, Down) is stored as a flat list of 9 elements.  
  - The full cube is thus a list of six 9-element lists:  
    ```prolog
    Cube = [Front, Right, Back, Left, Up, Down].
    ```

2. **Search Procedure**  
  - I used **IDDFS**, starting at depth 0 (no moves) and increasing up to a configurable maximum (default 4).  
  - At each depth *d*, a **depth-limited search** tries all sequences of *d* moves in turn.  
  - The search is **optimal** in the sense that it will find the shortest possible move sequence (if it exists within the depth limit).

3. **Move Generation & Pruning**  
  - I implemented 12 face-turn moves (`u, u′, d, d′, r, r′, l, l′, f, f′, b, b′`) plus 6 slice moves (`m, m′, e, e′, s, s′`), for a total of 18 primitive operations.  
  - To avoid immediately undoing the previous move, I filtered out the inverse of the last applied move at each step.  
  - The core search is orchestrated by the predicates `depth_limited_search/4` and `iterative_deepening_search/4`, with move application done by `move/3`.

> **Why IDDFS?**  
> IDDFS combines the space efficiency of DFS with the optimality guarantee of BFS up to the chosen depth limit. Given the exponential branching in Rubik’s-Cube move space, it’s a natural choice when an upper bound on solution length is known or can be reasonably estimated (rubik’s-Cube is always solvable within 20 moves).

---

## Usage

1. **Compile**  
  ```sh
  make
  ```
  This produces an executable `flp24-log`.

2. **Run**  
  ```sh
  ./flp24-log < tests/test1
  ```


## Test speed
  - test0: immediate
  - test1: under second
  - test2: ~3 seconds
  - for complicated inputs (>5 moves, the program runs too long and gets killed. I dont know if i have a mistake in some move and program gets into infinite loop, or if the algorithm is slow in general.)
