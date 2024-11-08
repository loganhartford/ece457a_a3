from q1 import *
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    population_size = 50
    num_generations = 150
    crossover_probs = [0.5, 0.6, 0.7]
    mutation_probs = [0.2, 0.25, 0.3]

    fig, axs = plt.subplots(len(crossover_probs), len(mutation_probs), figsize=(15, 15))
    fig.suptitle("Fitness Progression for Different Crossover and Mutation Probabilities", fontsize=16)

    for i, crossover_prob in enumerate(crossover_probs):
        for j, mutation_prob in enumerate(mutation_probs):
            print(f"Crossover: {crossover_prob}, Mutation: {mutation_prob}")
            start_time = time.time()
            best_individual, best_fitness, best_fitness_per_generation = genetic_algorithm(
                population_size, num_generations, crossover_prob, mutation_prob
            )
            end_time = time.time()

            print("Time taken:", end_time - start_time)
            print("Best Individual:", best_individual)
            print("Best Fitness:", best_fitness)

            # Plot results in the appropriate subplot
            axs[i, j].plot(best_fitness_per_generation)
            axs[i, j].set_title(f"Crossover: {crossover_prob}, Mutation: {mutation_prob}")
            axs[i, j].set_xlabel("Generation")
            axs[i, j].set_ylabel("Best Fitness")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
    plt.savefig("fitness_progress_grid_crossover_mutation.png")
    plt.show()
