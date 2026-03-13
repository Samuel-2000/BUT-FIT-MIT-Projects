from random import shuffle, seed as random_seed, randrange
import sys
from typing import Iterable, List, Optional, Tuple, Union, cast
import random
import math
from copy import deepcopy
import time

class UnsolvableSudoku(Exception):
    pass


class _SudokuSolver:
    def __init__(self, sudoku: 'Sudoku'):
        self.width = sudoku.width
        self.height = sudoku.height
        self.size = sudoku.size
        self.sudoku = sudoku

    def _solve(self) -> Optional['Sudoku']:
        blanks = self.__get_blanks()
        blank_count = len(blanks)
        are_blanks_filled = [False for _ in range(blank_count)]
        blank_fillers = self.__calculate_blank_cell_fillers(blanks)
        solution_board = self.__get_solution(
            Sudoku._copy_board(self.sudoku.board), blanks, blank_fillers, are_blanks_filled)
        solution_difficulty = 0
        if not solution_board:
            return None
        return Sudoku(self.width, self.height, board=solution_board, difficulty=solution_difficulty)

    def __calculate_blank_cell_fillers(self, blanks: List[Tuple[int, int]]) -> List[List[List[bool]]]:
        sudoku = self.sudoku
        valid_fillers = [[[True for _ in range(self.size)] for _ in range(
            self.size)] for _ in range(self.size)]
        for row, col in blanks:
            for i in range(self.size):
                same_row = sudoku.board[row][i]
                same_col = sudoku.board[i][col]
                if same_row and i != col:
                    valid_fillers[row][col][same_row - 1] = False
                if same_col and i != row:
                    valid_fillers[row][col][same_col - 1] = False
            grid_row, grid_col = row // sudoku.height, col // sudoku.width
            grid_row_start = grid_row * sudoku.height
            grid_col_start = grid_col * sudoku.width
            for y_offset in range(sudoku.height):
                for x_offset in range(sudoku.width):
                    if grid_row_start + y_offset == row and grid_col_start + x_offset == col:
                        continue
                    cell = sudoku.board[grid_row_start +
                                        y_offset][grid_col_start + x_offset]
                    if cell:
                        valid_fillers[row][col][cell - 1] = False
        return valid_fillers

    def __get_blanks(self) -> List[Tuple[int, int]]:
        blanks = []
        for i, row in enumerate(self.sudoku.board):
            for j, cell in enumerate(row):
                if cell == Sudoku._empty_cell_value:
                    blanks += [(i, j)]
        return blanks

    def __is_neighbor(self, blank1: Tuple[int, int], blank2: Tuple[int, int]) -> bool:
        row1, col1 = blank1
        row2, col2 = blank2
        if row1 == row2 or col1 == col2:
            return True
        grid_row1, grid_col1 = row1 // self.height, col1 // self.width
        grid_row2, grid_col2 = row2 // self.height, col2 // self.width
        return grid_row1 == grid_row2 and grid_col1 == grid_col2

    # Optimized version of above
    def __get_solution(self, board: List[List[Union[int, None]]], blanks: List[Tuple[int, int]], blank_fillers: List[List[List[bool]]], are_blanks_filled: List[bool]) -> Optional[List[List[int]]]:
        min_filler_count = None
        chosen_blank = None
        for i, blank in enumerate(blanks):
            x, y = blank
            if are_blanks_filled[i]:
                continue
            valid_filler_count = sum(blank_fillers[x][y])
            if valid_filler_count == 0:
                # Blank cannot be filled with any number, no solution
                return None
            if not min_filler_count or valid_filler_count < min_filler_count:
                min_filler_count = valid_filler_count
                chosen_blank = blank
                chosen_blank_index = i

        if not chosen_blank:
            # All blanks have been filled with valid values, return this board as the solution
            return cast(List[List[int]], board)

        row, col = chosen_blank

        # Declare chosen blank as filled
        are_blanks_filled[chosen_blank_index] = True

        # Save list of neighbors affected by the filling of current cell
        revert_list = [False for _ in range(len(blanks))]

        for number in range(self.size):
            # Only try filling this cell with numbers its neighbors aren't already filled with
            if not blank_fillers[row][col][number]:
                continue

            # Test number in this cell, number + 1 is used because number is zero-indexed
            board[row][col] = number + 1

            for i, blank in enumerate(blanks):
                blank_row, blank_col = blank
                if blank == chosen_blank:
                    continue
                if self.__is_neighbor(blank, chosen_blank) and blank_fillers[blank_row][blank_col][number]:
                    blank_fillers[blank_row][blank_col][number] = False
                    revert_list[i] = True
                else:
                    revert_list[i] = False
            solution_board = self.__get_solution(
                board, blanks, blank_fillers, are_blanks_filled)

            if solution_board:
                return solution_board

            # No solution found by having tested number in this cell
            # So we reallow neighbor cells to have this number filled in them
            for i, blank in enumerate(blanks):
                if revert_list[i]:
                    blank_row, blank_col = blank
                    blank_fillers[blank_row][blank_col][number] = True

        # If this point is reached, there is no solution with the initial board state,
        # a mistake must have been made in earlier steps

        # Declare chosen cell as empty once again
        are_blanks_filled[chosen_blank_index] = False
        board[row][col] = Sudoku._empty_cell_value

        return None










def linear(initial_temperature, min_temperature, max_iterations, iteration):
    return initial_temperature - ((initial_temperature - min_temperature) / max_iterations) * iteration

def logarithmic(initial_temperature, iteration):
    return initial_temperature / math.log(iteration + 1)

def geometric(temperature, cooling_factor):
    return temperature * cooling_factor

def exponential(initial_temperature, max_iterations, iteration):
    return initial_temperature * math.exp(-iteration / max_iterations)

def calculate_geometric_factor(initial_temperature, min_temperature, max_iterations):
    return (min_temperature / initial_temperature) ** (1 / max_iterations)

class SimulatedAnnealingSolver:
    def __init__(self, sudoku: 'Sudoku'):
        self.sudoku = sudoku
        self.mutable_cells = [[False for _ in range(sudoku.size)] for _ in range(sudoku.size)]
        self.squares_with_two_mutables = []
        self.mutable_cells_by_square = {}


    def solve(self, cooling_function,  max_iterations, adaptive, initial_temperature = 100.0, min_temperature = 0.1):
        current_solution = self.sudoku

        #print("init solution:")
        #print(current_solution)

        self.fill_empty_cells_randomly(current_solution)
        self.init_squares_with_two_mutables_indices() # algorithm will know, which squares to ignore (after initialisation only 1 or 0 empty cells.)
        current_fitness = self.calculate_fitness(current_solution)

        if adaptive:
            best_solution = current_solution

        best_fitness = current_fitness

        #print("start solution:")
        #print(current_solution)
        # print(f"start fitness: {best_fitness}")   

        #start_time = time.time()
        last_update_iteration = 0
        finished = 0
        if cooling_function == "geometric":
            cooling_factor = calculate_geometric_factor(initial_temperature, min_temperature, max_iterations)

        temperature = initial_temperature

        iteration_fitness_pairs = [(0, best_fitness)]  # Store (iteration, best_fitness) pairs.

        for iteration in range(1, max_iterations):
            if current_fitness == 0:
                finished = 1
                break

            if cooling_function == "linear": 
                temperature = linear(initial_temperature, min_temperature, max_iterations, iteration)
            elif cooling_function == "logarithmic":
                temperature = logarithmic(initial_temperature, iteration)
            elif cooling_function == "geometric":
                temperature = geometric(temperature, cooling_factor)
            else: # cooling_function == "exponential":
                temperature = exponential(initial_temperature, max_iterations, iteration)

            if adaptive and iteration - last_update_iteration > 100 * best_fitness:
                current_solution = best_solution
                last_update_iteration = iteration
            

            child_solution = self.get_mutated_random_cell(current_solution)
            child_fitness = self.calculate_fitness(child_solution)

            diff = current_fitness - child_fitness

            # calculate metropolis acceptance criterion
            metropolis = math.exp(diff / temperature)

            if diff > 0 or metropolis > random.random():
                current_solution = child_solution
                current_fitness = child_fitness
                #if diff <= 0:
                #    print(f"[> 0]: {metropolis}")


            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness
                #print(f"best fitness: {best_fitness} \t Time taken: {time.time() - start_time} seconds \t iteration: {iteration} \t at temperature: {temperature}") 
                last_update_iteration = iteration
                iteration_fitness_pairs.append((iteration, best_fitness))

        #if (best_fitness == 0):
        #    print(best_solution)
        # print("\n____________________________________________\n")
        return (finished, last_update_iteration), iteration_fitness_pairs


    def fill_empty_cells_randomly(self, sudoku):
        square_values = set(range(1, sudoku.size + 1)) #set(list(range(1, self.size + 1)))

        for row_start in range(0, sudoku.size, sudoku.height):
            for col_start in range(0, sudoku.size, sudoku.width):
                used_values = set(sudoku.board[r][c]
                    for r in range(row_start, row_start + sudoku.height)
                    for c in range(col_start, col_start + sudoku.width)
                )
                
                if None not in used_values:
                    continue
                
                available_values = list(square_values - used_values)
                shuffle(available_values)
                
                for row in range(row_start, row_start + sudoku.height):
                    for col in range(col_start, col_start + sudoku.width):
                        if sudoku.board[row][col] == Sudoku._empty_cell_value:
                            sudoku.board[row][col] = available_values.pop()
                            self.mutable_cells[row][col] = True


    def init_squares_with_two_mutables_indices(self):
        for square_row in range(self.sudoku.size // self.sudoku.height):
            for square_col in range(self.sudoku.size // self.sudoku.width):
                row_start = square_row * self.sudoku.height
                col_start = square_col * self.sudoku.width
                
                mutable_count = sum(self.mutable_cells[row][col] for row in range(row_start, row_start + self.sudoku.height)
                                for col in range(col_start, col_start + self.sudoku.width))

                if mutable_count >= 2:
                    self.squares_with_two_mutables.append((square_row, square_col))
                    self.mutable_cells_by_square[square_row, square_col] = [(row, col) for row in range(row_start, row_start + self.sudoku.height)
                                    for col in range(col_start, col_start + self.sudoku.width)
                                    if self.mutable_cells[row][col]]


    def get_mutated_random_cell(self, sudoku: 'Sudoku') -> 'Sudoku':
        cells_list = self.mutable_cells_by_square[random.choice(self.squares_with_two_mutables)]
        shuffle(cells_list)
        cell1_row, cell1_col = cells_list[0]
        cell2_row, cell2_col = cells_list[1]

        # Swap the values in the selected cells
        new_sudoku = sudoku
        sudoku.board[cell1_row][cell1_col], new_sudoku.board[cell2_row][cell2_col] = sudoku.board[cell2_row][cell2_col], new_sudoku.board[cell1_row][cell1_col]
        
        return new_sudoku
    
    

    def calculate_fitness(self, sudoku: 'Sudoku') -> float:
        def count_non_unique(seq: Iterable[int]) -> int:
            unique_values = set()
            non_unique_count = 0
            for cell in seq:
                if cell in unique_values:
                    non_unique_count += 1
                else:
                    unique_values.add(cell)
            return non_unique_count

        non_unique_count = 0

        # Check rows and columns for non-unique values
        for i in range(sudoku.size):
            non_unique_count += count_non_unique(sudoku.board[i])  # Rows
            non_unique_count += count_non_unique(sudoku.board[j][i] for j in range(sudoku.size))  # Columns

        return non_unique_count
    

class Sudoku:
    _empty_cell_value = None

    def __init__(self, width: int = 3, height: Optional[int] = None, board: Optional[Iterable[Iterable[Union[int, None]]]] = None, difficulty: Optional[float] = None, seed: int = randrange(sys.maxsize)):
        """
        Initializes a Sudoku board

        :param width: Integer representing the width of the Sudoku grid. Defaults to 3.
        :param height: Optional integer representing the height of the Sudoku grid. If not provided, defaults to the value of `width`.
        :param board: Optional iterable for a the initial state of the Sudoku board.
        :param difficulty: Optional float representing the difficulty level of the Sudoku puzzle. If provided, sets the difficulty level based on the number of empty cells. Defaults to None.
        :param seed: Integer representing the seed for the random number generator used to generate the board. Defaults to a random seed within the system's maximum size.

        :raises AssertionError: If the width, height, or size of the board is invalid.
        """
        self.width = width
        self.height = height if height else width
        self.size = self.width * self.height
        self.__difficulty: float

        assert self.width > 0, 'Width cannot be less than 1'
        assert self.height > 0, 'Height cannot be less than 1'
        assert self.size > 1, 'Board size cannot be 1 x 1'

        if difficulty is not None:
            self.__difficulty = difficulty

        if board:
            blank_count = 0
            self.board: List[List[Union[int, None]]] = [
                [cell for cell in row] for row in board]
            for row in self.board:
                for i in range(len(row)):
                    if not row[i] in range(1, self.size + 1):
                        row[i] = Sudoku._empty_cell_value
                        blank_count += 1
            if difficulty == None:
                if self.validate():
                    self.__difficulty = blank_count / \
                        (self.size * self.size)
                else:
                    self.__difficulty = -2
        else:
            positions = list(range(self.size))
            random_seed(seed)
            shuffle(positions)
            self.board = [[(i + 1) if i == positions[j]
                           else Sudoku._empty_cell_value for i in range(self.size)] for j in range(self.size)]
            
    def solve_with_simulated_annealing(self, cooling_function, max_iterations, adaptive):
        solver = SimulatedAnnealingSolver(self)
        return solver.solve(cooling_function=cooling_function, max_iterations=max_iterations, adaptive=adaptive)

    

    def solve(self, raising: bool = False) -> 'Sudoku':
        """
        Solves the given Sudoku board

        :param raising: Boolean for if you wish to raise an UnsolvableSodoku error when the board is invalid. Defaults to `false`.
        :raises UnsolvableSudoku:
        """
        solution = _SudokuSolver(self)._solve() if self.validate() else None
        if solution:
            return solution
        elif raising:
            raise UnsolvableSudoku('No solution found')
        else:
            solution_board = Sudoku.empty(self.width, self.height).board
            solution_difficulty = -2
            return Sudoku(board=solution_board, difficulty=solution_difficulty)

    def validate(self) -> bool:
        row_numbers = [[False for _ in range(self.size)]
                       for _ in range(self.size)]
        col_numbers = [[False for _ in range(self.size)]
                       for _ in range(self.size)]
        box_numbers = [[False for _ in range(self.size)]
                       for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                cell = self.board[row][col]
                box = (row // self.height) * self.height + (col // self.width)
                if cell == Sudoku._empty_cell_value:
                    continue
                elif isinstance(cell, int):
                    if row_numbers[row][cell - 1]:
                        return False
                    elif col_numbers[col][cell - 1]:
                        return False
                    elif box_numbers[box][cell - 1]:
                        return False
                    row_numbers[row][cell - 1] = True
                    col_numbers[col][cell - 1] = True
                    box_numbers[box][cell - 1] = True
        return True

    @ staticmethod
    def _copy_board(board: Iterable[Iterable[Union[int, None]]]) -> List[List[Union[int, None]]]:
        return [[cell for cell in row] for row in board]

    @ staticmethod
    def empty(width: int, height: int):
        size = width * height
        board = [[Sudoku._empty_cell_value] * size] * size
        return Sudoku(width, height, board, 0)

    def difficulty(self, difficulty: float) -> 'Sudoku':
        """
        Sets the difficulty of the Sudoku board by removing cells.

        This method modifies the current Sudoku instance by removing cells from the solved puzzle to achieve the desired difficulty level. The difficulty is specified as a float value between 0 and 1, where 0 represents the easiest puzzle (fully solved) and 1 represents the most difficult puzzle (almost empty).

        :param difficulty: A float value between 0 and 1 representing the desired difficulty level of the Sudoku puzzle.
        :return: A new Sudoku instance representing the puzzle with adjusted difficulty.
        :raises AssertionError: If the provided difficulty value is not within the range of 0 to 1.
        """
        assert 0 < difficulty < 1, 'Difficulty must be between 0 and 1'
        indices = list(range(self.size * self.size))
        shuffle(indices)
        problem_board = self.solve().board
        for index in indices[:int(difficulty * self.size * self.size)]:
            row_index = index // self.size
            col_index = index % self.size
            problem_board[row_index][col_index] = Sudoku._empty_cell_value
        return Sudoku(self.width, self.height, problem_board, difficulty)

    def show(self) -> None:
        """
        Prints the puzzle to the terminal
        """
        if self.__difficulty == -2:
            print('Puzzle has no solution')
        if self.__difficulty == -1:
            print('Invalid puzzle. Please solve the puzzle (puzzle.solve()), or set a difficulty (puzzle.difficulty())')
        if not self.board:
            print('No solution')
        print(self.__format_board_ascii())

    def show_full(self) -> None:
        """
        Prints the puzzle to the terminal, with more information
        """
        print(self.__str__())

    def __format_board_ascii(self) -> str:
        table = ''
        cell_length = len(str(self.size))
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(self.board):
            if i == 0:
                table += ('+-' + '-' * (cell_length + 1) *
                          self.width) * self.height + '+' + '\n'
            table += (('| ' + '{} ' * self.width) * self.height + '|').format(*[format_int.format(
                x) if x != Sudoku._empty_cell_value else ' ' * cell_length for x in row]) + '\n'
            if i == self.size - 1 or i % self.height == self.height - 1:
                table += ('+-' + '-' * (cell_length + 1) *
                          self.width) * self.height + '+' + '\n'
        return table

    def __str__(self) -> str:
        if self.__difficulty == -2:
            difficulty_str = 'INVALID PUZZLE (GIVEN PUZZLE HAS NO SOLUTION)'
        elif self.__difficulty == -1:
            difficulty_str = 'INVALID PUZZLE'
        elif self.__difficulty == 0:
            difficulty_str = 'SOLVED'
        else:
            difficulty_str = '{:.2f}'.format(self.__difficulty)
        return '''
---------------------------
{}x{} ({}x{}) SUDOKU PUZZLE
Difficulty: {}
---------------------------
{}
        '''.format(self.size, self.size, self.width, self.height, difficulty_str, self.__format_board_ascii())





