
3. **Solver Selection:** By default, the program uses a restarting greedy 1-path search strategy (`dummy`). You can also choose from other available solvers using the `--solver` flag:

   - `--solver dummy`: Restarting greedy 1-path search.
   - `--solver bfs`: Breadth-first search.
   - `--solver dfs`: Depth-first search (with an optional depth limit using `--dls-limit DEPTH`).
   - `--solver a_star`: A* search (with heuristic selection, e.g., `--heuristic nb_not_home` or `--heuristic student`).

4. **Deal Difficulty:** You can control the difficulty of card deals using the `--easy-mode` flag, which specifies the maximum number of reverse moves made in the initial deal setup.

5. **Memory Limit:** To limit memory usage, you can use the `--mem-limit` flag, which sets the maximum memory consumption in bytes. If the program exceeds this limit, it will abort itself.




In FreeCell, finding an optimal heuristic that always underestimates the true cost can be challenging due to the complex nature of the game. However, you can experiment with different heuristics to see which one works best for your specific scenarios. Here are a few additional heuristic ideas that you can explore:

1. **Empty Columns and Card Sequence Heuristic:**
   - Count the number of empty columns.
   - Analyze the sequences of cards in the columns and calculate how many cards are in sequence (e.g., Ace, 2, 3, etc.).
   - Add the count of empty columns and the count of in-sequence cards to get the heuristic value.
   - This heuristic takes into account both the empty columns and the potential for card sequences, providing a more informed estimate.

2. **Advanced Card Positioning Heuristic:**
   - Analyze the positions of cards in the columns.
   - Penalize cards that are blocking other cards from being moved or are in positions where they are challenging to move.
   - Reward cards that are closer to their home cells or in positions where they can easily be moved to their home cells.
   - Consider the number of empty columns as well.
   - This heuristic would require more complex analysis of card positions but could provide better guidance.

3. **Combination of Heuristics:**
   - Use a combination of heuristics based on the current state. For example, you could switch between different heuristics based on the number of empty columns or the presence of certain card sequences.
   - This approach allows adaptability and can lead to better heuristic estimates depending on the game's characteristics.

Keep in mind that finding the best heuristic often involves a balance between accuracy and computational efficiency. You may need to experiment with different heuristics and fine-tune them to achieve the best performance on a variety of FreeCell scenarios. Additionally, heuristics can be domain-specific, so the effectiveness of a heuristic may vary depending on the distribution of FreeCell game states you encounter.

