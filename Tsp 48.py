import random
import tsplib95
import matplotlib.pyplot as plt

# Load TSP problem using TSPLIB library
def load_tsp_problem(file_path):
    problem = tsplib95.load(file_path)
    return problem

# Initialize a population of random chromosomes
def initialize_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        chromosome = list(range(1, num_cities + 1))
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

# Calculate the total distance of a route
def calculate_total_distance(route, problem):
    total_distance = 0
    for i in range(len(route)):
        total_distance += problem.get_weight(route[i], route[(i + 1) % len(route)])
    return total_distance

# Tournament selection
def tournament_selection(population, problem, tournament_size):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: calculate_total_distance(x, problem))

# Order crossover (OX)
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [gene for gene in parent2 if gene not in child]
    remaining_idx = 0
    for i in range(len(parent2)):
        if child[i] == -1:
            child[i] = remaining[remaining_idx]
            remaining_idx += 1

    return child

# PMX crossover
def pmx_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end + 1] = parent1[start:end + 1]

    for i in range(len(parent2)):
        if child[i] == -1:
            current_gene = parent2[i]
            while current_gene in child:
                current_gene = parent2[child.index(current_gene)]
            child[i] = current_gene

    return child

# Swap mutation
def swap_mutation(chromosome):
    idx1, idx2 = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

# Inversion mutation
def inversion_mutation(chromosome):
    start, end = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[start:end + 1] = reversed(chromosome[start:end + 1])
    return chromosome

# Genetic algorithm
def genetic_algorithm(problem, population_size, generations, crossover_rate, mutation_rate, tournament_size):
    num_cities = problem.dimension
    population = initialize_population(population_size, num_cities)
    best_solution = min(population, key=lambda x: calculate_total_distance(x, problem))
    best_distances = [calculate_total_distance(best_solution, problem)]
    stationary_count = 0
    stationary_threshold = 50  # Adjust as needed

    for generation in range(generations):
        new_population = []

        # Elitism: Preserve the best solution
        new_population.append(best_solution)

        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, problem, tournament_size)
            parent2 = tournament_selection(population, problem, tournament_size)

            # Crossover
            if random.random() < crossover_rate:
                if random.random() < 0.5:
                    child = order_crossover(parent1, parent2)
                else:
                    child = pmx_crossover(parent1, parent2)
            else:
                child = parent1

            # Mutation
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    child = swap_mutation(child)
                else:
                    child = inversion_mutation(child)

            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda x: calculate_total_distance(x, problem))
        
        # Check for stationary state
        if current_best == best_solution:
            stationary_count += 1
        else:
            stationary_count = 0

        # Update best solution
        if calculate_total_distance(current_best, problem) < calculate_total_distance(best_solution, problem):
            best_solution = current_best

        best_distances.append(calculate_total_distance(best_solution, problem))

        # Check for stationary state
        if stationary_count >= stationary_threshold:
            print(f"Reached a stationary state after {generation} generations.")
            break

    return best_solution, best_distances

# Function to plot evolution of distances
def plot_evolution(best_distances):
    plt.plot(best_distances)
    plt.xlabel('Generation')
    plt.ylabel('Total Distance')
    plt.title('Evolution of Total Distance in Genetic Algorithm')
    plt.show()

# Function to run the algorithm with different parameter combinations
def run_algorithm_with_parameters(problem, population_size, generations, crossover_rate, mutation_rate, tournament_size):
    best_solution, best_distances = genetic_algorithm(problem, population_size, generations, crossover_rate, mutation_rate, tournament_size)
    
    print(f"Best solution: {best_solution}")
    print(f"Total distance: {calculate_total_distance(best_solution, problem)}")

    # Plot the evolution of distances
    plot_evolution(best_distances)

# Example usage
if __name__ == "__main__":
    # Modify the file paths based on your datasets
    files = [ 
             'dataset/48.tsp']

    for file_path in files:
        tsp_problem = load_tsp_problem(file_path)

        print(f"\nRunning for problem: {file_path}")
        # Run the algorithm with different parameter combinations
        run_algorithm_with_parameters(tsp_problem, population_size=100, generations=1000, crossover_rate=0.8, mutation_rate=0.1, tournament_size=5)
