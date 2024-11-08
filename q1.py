import numpy as np
import control as ctrl
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time

Kp_range = (2, 18)
Ti_range = (1.05, 9.42)
Td_range = (0.26, 2.37)

def random_individual():
    return np.array(
        [
            round(np.random.uniform(*Kp_range), 2),
            round(np.random.uniform(*Ti_range), 2),
            round(np.random.uniform(*Td_range), 2)
        ])

def is_stable(system):
    poles = ctrl.pole(system)
    return np.all(np.real(poles) < 0)

def compute_step_metrics(t, y):
    y_final = y[-1]
    y_10 = y_final * 0.1
    y_90 = y_final * 0.9

    # Rise time
    try:
        idx_10 = np.where(y >= y_10)[0][0]
        idx_90 = np.where(y >= y_90)[0][0]
        t_r = t[idx_90] - t[idx_10]
    except IndexError:
        t_r = float('inf')

    # Settling time (2% criterion)
    within_2_percent = np.abs(y - y_final) <= 0.02 * y_final
    if np.any(within_2_percent):
        t_s_candidates = t[within_2_percent]
        t_s = t_s_candidates[0]
    else:
        t_s = float('inf')

    # Overshoot
    M_p = (np.max(y) - y_final) / y_final * 100 if y_final != 0 else float('inf')

    return t_r, t_s, M_p

def performance_metrics(Kp, Ti, Td):
    G = Kp * ctrl.TransferFunction([Ti * Td, Ti, 1], [Ti, 0])
    F = ctrl.TransferFunction([1], [1, 6, 11, 6, 0])
    sys = ctrl.feedback(ctrl.series(G, F), 1)
    
    # Check if the system is stable
    if not is_stable(sys):
        return float('inf'), float('inf'), float('inf'), float('inf')
    
    t = np.linspace(0, 20, num=2001)
    t, y = ctrl.step_response(sys, t)
    
    # Calculate performance metrics
    ISE = np.trapz((y - 1)**2, t)
    t_r, t_s, M_p = compute_step_metrics(t, y)

    # Handle NaN values by returning a high penalty
    if np.isnan(ISE) or np.isnan(t_r) or np.isnan(t_s) or np.isnan(M_p):
        return float('inf'), float('inf'), float('inf'), float('inf')
    
    return ISE, t_r, t_s, M_p

def fitness_function(individual):
    try:
        ISE, t_r, t_s, M_p = performance_metrics(*individual)
        return ISE + t_r + t_s + M_p
    except Exception:
        return float('inf')

def initialize_population(population_size):
    population = []
    fitnesses = []
    while len(population) < population_size:
        individual = random_individual()
        fitness = fitness_function(individual)
        if np.isfinite(fitness):
            population.append(individual)
            fitnesses.append(fitness)
    population = np.array(population)
    fitnesses = np.array(fitnesses)
    return population, fitnesses

def select_parents(population, selection_probs):
    parents_idx = np.random.choice(len(population), size=2, p=selection_probs)
    return population[parents_idx[0]], population[parents_idx[1]]

def crossover(parent1, parent2):
    if np.random.random() < crossover_prob:
        # Single point crossover
        crossover_point = np.random.randint(1, 3)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual):
    if np.random.random() < mutation_prob:
        # Gaussian mutation
        individual[0] = round(np.clip(individual[0] + np.random.normal(0, 1), *Kp_range), 2)
        individual[1] = round(np.clip(individual[1] + np.random.normal(0, 1), *Ti_range), 2)
        individual[2] = round(np.clip(individual[2] + np.random.normal(0, 1), *Td_range), 2)
    return individual

def compute_fitnesses(population):
    with Pool() as pool:
        fitnesses = pool.map(fitness_function, [individual for individual in population])
    return np.array(fitnesses)

def filter_finite_individuals(population, fitnesses):
    finite_indices = np.isfinite(fitnesses)
    return population[finite_indices], fitnesses[finite_indices]

def compute_selection_probs(fitnesses):
    epsilon = 1e-6
    scaled_fitnesses = fitnesses + epsilon
    inverse_fitnesses = 1 / scaled_fitnesses
    total = np.sum(inverse_fitnesses)
    if total == 0:
        selection_probs = np.ones_like(fitnesses) / len(fitnesses)
    else:
        selection_probs = inverse_fitnesses / total
    return selection_probs

def genetic_algorithm(population_size, num_generations):
    population, fitnesses = initialize_population(population_size)
    best_fitness_per_generation = []
    best_individual = None
    best_fitness = float('inf')
    
    for generation in range(num_generations):
        print(f'Generation {generation + 1}/{num_generations}')
        
        # Evaluate fitnesses
        if generation > 0:
            fitnesses = compute_fitnesses(population)
        
        # Filter out individuals with infinite fitness
        population, fitnesses = filter_finite_individuals(population, fitnesses)
        
        if len(population) == 0:
            # Reinitialize population if all individuals are invalid
            population, fitnesses = initialize_population(population_size)
        
        # Update best individual
        min_fitness_idx = np.argmin(fitnesses)
        if fitnesses[min_fitness_idx] < best_fitness:
            best_fitness = fitnesses[min_fitness_idx]
            best_individual = population[min_fitness_idx]
        best_fitness_per_generation.append(best_fitness)
        
        # Elitism: Keep the best two individuals
        sorted_indices = np.argsort(fitnesses)
        next_population = population[sorted_indices[:2]].copy()
        
        # Compute selection probabilities
        selection_probs = compute_selection_probs(fitnesses)
        
        # Generate new offspring
        while len(next_population) < population_size:
            parent1, parent2 = select_parents(population, selection_probs)
            child1, child2 = crossover(parent1, parent2)
    
            child1 = mutate(child1)
            child2 = mutate(child2)
    
            fitness1 = fitness_function(child1)
            fitness2 = fitness_function(child2)
    
            # Only add children with finite fitness
            if np.isfinite(fitness1):
                next_population = np.vstack([next_population, child1])
            if len(next_population) < population_size and np.isfinite(fitness2):
                next_population = np.vstack([next_population, child2])
        
        population = next_population[:population_size]
    
    return best_individual, best_fitness, best_fitness_per_generation

def plot_results(best_fitness_per_generation, filename='fitness_progress.png'):
    plt.figure()
    plt.plot(best_fitness_per_generation)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Optimization Progress")
    plt.savefig(filename)  # Save the figure to a file
    plt.show()

def test_1_3():
    start_time = time.time()
    best_individual, best_fitness, best_fitness_per_generation = genetic_algorithm(population_size, num_generations)
    end_time = time.time()

    print("Time taken:", end_time - start_time)
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_fitness)
    plot_results(best_fitness_per_generation, filename='1.3.png')

if __name__ == '__main__':
    population_size = 50
    num_generations = 150
    crossover_prob = 0.6
    mutation_prob = 0.25
    test_1_3()