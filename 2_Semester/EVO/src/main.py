from sudoku import Sudoku
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import seaborn as sns
import numpy as np

def run_tests(cooling, difficulties, boxplots_filename, lineplots_filename, adaptive, test_count=100, max_iterations=100000):
    pdf_pages_box = PdfPages(boxplots_filename)
    _, axs_box = plt.subplots(len(difficulties), 1, figsize=(len(cooling)*2, len(difficulties)*6))

    # For line plots with deviations
    pdf_pages_line = PdfPages(lineplots_filename)
    _, axs_line = plt.subplots(len(difficulties), len(cooling), figsize=(len(cooling)*4, len(difficulties)*3))  # Adjusted dimensions

    for i, difficulty in enumerate(tqdm(difficulties, desc="Difficulty")):
        all_iterations = []  # to store all iterations for this difficulty level
        boxplot_sum_finished = []
        iterations_taken = []  # Reset for next difficulty level
        for j, cooling_function in enumerate(tqdm(cooling, desc=f'Cooling for Difficulty: {difficulty}', leave=False)):
            puzzles = [Sudoku(3).difficulty(difficulty) for _ in range(test_count)]
            iterations = []  # Separate list for each cooling function.
            iteration_fitness_pairs_all = []
            sum_finished = 0
            for puzzle in tqdm(puzzles, desc=f'Puzzles for Cooling: {cooling_function}', leave=False):
                (finished, iteration), iteration_fitness_pairs = puzzle.solve_with_simulated_annealing(cooling_function, max_iterations, adaptive)
                if finished:
                    sum_finished += finished
                    iterations.append(iteration)  # Collect iterations for this cooling function
                    all_iterations.extend(iterations)  # Extend all_iterations list
                    iteration_fitness_pairs_all.append(iteration_fitness_pairs)
            
            iterations_taken.append(iterations)  # Collect iterations for all cooling functions
            boxplot_sum_finished.append(sum_finished)

            # Plotting line plots with deviations
            for k, iteration_fitness_pairs in enumerate(iteration_fitness_pairs_all):
                iteration_fitness_pairs = np.array(iteration_fitness_pairs)
                if iteration_fitness_pairs.size > 0:
                    iterations = iteration_fitness_pairs[:, 0]
                    fitness = iteration_fitness_pairs[:, 1]
                    sns.lineplot(x=iterations, y=fitness, err_style="band", err_kws={'alpha':0.3}, ax=axs_line[i, j])
                    axs_line[i, j].set_xscale('log')  # Set logarithmic scale for x-axis

        boxplot_labels = [f"{cooling_function} ({sum_finished})" for cooling_function, sum_finished in zip(cooling, boxplot_sum_finished)]

        for j, cooling_function in enumerate(cooling):
            axs_line[i, j].set_title(f'Difficulty: {difficulty}, Cooling: {cooling_function}')

        axs_box[i].boxplot(iterations_taken, labels=boxplot_labels)
        axs_box[i].set_title(f'Difficulty: {difficulty}')
        axs_box[i].set_xlabel(f'Cooling Function (Samples Finished out of {test_count})')
        axs_box[i].set_ylabel('Iterations')
    
        if len(all_iterations) == 0:
            axs_box[i].set_ylim(0, max_iterations)
        else: # Set y-axis limits based on the minimum and maximum iterations observed
            axs_box[i].set_ylim(0, min(max(all_iterations)*1.2, max_iterations))
        axs_box[i].grid(True)

    plt.tight_layout()
    pdf_pages_line.savefig()
    plt.close()
    pdf_pages_line.close()

    plt.tight_layout()
    pdf_pages_box.savefig()
    plt.close()
    pdf_pages_box.close()



if __name__ == "__main__":
    program_start_time = time.time()

    cooling = ["geometric", "logarithmic", "linear", "exponential"]
    max_iterations = 100000
    test_count = 100
    difficulties = [0.25, 0.28, 0.3]

    run_tests(cooling, difficulties, "box_plots.pdf",           "line_plots.pdf",           adaptive=False, test_count=test_count, max_iterations=max_iterations)
    #run_tests(cooling, difficulties, "box_plots_adaptive.pdf",  "line_plots_adaptive.pdf",  adaptive=True,  test_count=test_count, max_iterations=max_iterations)

    program_end_time = time.time()
    print(f"program execution time: {program_end_time - program_start_time}")
















