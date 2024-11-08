from q1 import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    num_generations = [5, 20, 50]
    num_populations = [5, 20, 50]

    fig, axs = plt.subplots(len(num_generations), len(num_populations), figsize=(15, 15))
    fig.suptitle("Fitness Progression for Different Population Sizes and Generations", fontsize=16)

    for i, num_generation in enumerate(num_generations):
        for j, num_population in enumerate(num_populations):
            start_time = time.time()
            best_individual, best_fitness, best_fitness_per_generation = genetic_algorithm(num_population, num_generation)
            end_time = time.time()

            print(f"Population Size: {num_population}, Number of Generations: {num_generation}")
            print("Time taken:", end_time - start_time)
            print("Best Individual:", best_individual)
            print("Best Fitness:", best_fitness)
            
            axs[i, j].plot(best_fitness_per_generation)
            axs[i, j].set_title(f"Pop: {num_population}, Gen: {num_generation}")
            axs[i, j].set_xlabel("Generation")
            axs[i, j].set_ylabel("Best Fitness")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fitness_progress_grid.png")
    plt.show()
