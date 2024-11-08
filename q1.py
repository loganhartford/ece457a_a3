import numpy as np

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
import control as ctrl

def is_stable(system):
    poles = ctrl.pole(system)
    return np.all(np.real(poles) < 0)

def performance_metrics(Kp, Ti, Td):
    G = Kp * ctrl.TransferFunction([Ti * Td, Ti, 1], [Ti, 0])
    F = ctrl.TransferFunction([1], [1, 6, 11, 6, 0])
    sys = ctrl.feedback(ctrl.series(G, F), 1)
    
    # Check if the system is stable
    if not is_stable(sys):
        # Return a high penalty if the system is unstable
        return float('inf'), float('inf'), float('inf'), float('inf')
    
    t = np.linspace(0, 100, num=10001)
    t, y = ctrl.step_response(sys, t)
    
    # Calculate performance metrics
    ISE = np.sum((y - 1)**2)
    sys_info = ctrl.step_info(sys)
    t_r = sys_info.get('RiseTime', float('inf'))
    t_s = sys_info.get('SettlingTime', float('inf'))
    M_p = sys_info.get('Overshoot', float('inf'))
    
    # Handle NaN values by returning a high penalty
    if np.isnan(ISE) or np.isnan(t_r) or np.isnan(t_s) or np.isnan(M_p):
        return float('inf'), float('inf'), float('inf'), float('inf')
    
    return ISE, t_r, t_s, M_p

def fitness_function(individual):
    ISE, t_r, t_s, M_p = performance_metrics(*individual)
    return ISE + t_r + t_s + M_p 

population_size = 50
num_generations = 150
crossover_prob = 0.6
mutation_prob = 0.25
def initialize_population(population_size):
    i = 0
    population = []
    fitnesses = []
    while i < population_size:
        individual = random_individual()
        fitness = fitness_function(individual)

        # Unstable solutions break the algoirthm, so we ignore them
        if not np.isinf(fitness):
            population.append(individual)
            fitnesses.append(fitness)
            i += 1
    population = np.array(population)
    fitnesses = np.array(fitnesses)
    return population, fitnesses

def select_parents(population, selection_probs):
    parents_idx = np.random.choice(population_size, size=2, p=selection_probs)
    return population[parents_idx[0]], population[parents_idx[1]]

def crossover(parent1, parent2):
    if np.random.random() < crossover_prob:
        # Single point crossover
        crossover_point = np.random.randint(1, 3)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    return parent1, parent2

def mutate(individual):
    if np.random.random() < mutation_prob:
        # Guassian mutation
        individual[0] = round(np.clip(individual[0] + np.random.normal(0, 1), *Kp_range), 2)
        individual[1] = round(np.clip(individual[1] + np.random.normal(0, 1), *Ti_range), 2)
        individual[2] = round(np.clip(individual[2] + np.random.normal(0, 1), *Td_range), 2)
    return individual

def genetic_algorithm(population_size, num_generations):
    population, fitnesses = initialize_population(population_size)
    # print(population)?
    best_fitness_per_generation = []
    
    for generation in range(num_generations):
        print(f'Generation {generation + 1}/{num_generations}')
        if generation > 0:
            fitnesses = np.array([fitness_function(individual) for individual in population])

        best_fitness = min(fitnesses)
        best_fitness_per_generation.append(best_fitness)
        
        # Elitism: Keep the best two individuals
        sorted_indices = np.argsort(fitnesses)
        next_population = population[sorted_indices[:2]]
        next_fitnesses = fitnesses[sorted_indices[:2]]

        # Generate new offspring
        selection_probs = fitnesses / np.sum(fitnesses)
        while len(next_population) < population_size:
            parent1, parent2 = select_parents(population, selection_probs)
            child1, child2 = crossover(parent1, parent2)
    
            child1 = mutate(child1)
            child2 = mutate(child2)

            fitness1 = fitness_function(child1)
            fitness2 = fitness_function(child2)

            # Unstable solutions break the algorithm, so we ignore them
            if not np.isinf(fitness1):
                next_population = np.vstack([next_population, child1])
                next_fitnesses = np.append(next_fitnesses, fitness1)
            if not np.isinf(fitness2):
                next_population = np.vstack([next_population, child2])
                next_fitnesses = np.append(next_fitnesses, fitness2)
        
        population = next_population[:population_size]
    
    best_individual = population[sorted_indices[0]]
    return best_individual, best_fitness_per_generation