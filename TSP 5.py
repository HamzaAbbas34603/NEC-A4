import numpy as np
import random
import matplotlib.pyplot as plt

class TSPProblem:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)

    def get_weight(self, city1, city2):
        return self.distance_matrix[city1][city2]

def load_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[1:-1]  # Skip the first & last lines
        distance_matrix = [list(map(float, line.split())) for line in lines]

    # Ensure the matrix is square
    num_cities = len(distance_matrix)
    for row in distance_matrix:
        if len(row) != num_cities:
            raise ValueError("Invalid distance matrix. It should be a square matrix.")

    return np.array(distance_matrix)

def calculate_total_distance(route, problem):
    total_distance = 0
    for i in range(len(route)):
        total_distance += problem.get_weight(route[i], route[(i + 1) % len(route)])
    return total_distance

def generate_random_route(num_cities):
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def initialize_population(population_size, num_cities):
    population = []
    for _ in range(population_size):
        chromosome = list(range(num_cities))
        random.shuffle(chromosome)
        population.append(chromosome)
    return population

def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i-1]][route[i]]
    return total_distance

def tournament_selection(population, distance_matrix, tournament_size):
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: calculate_total_distance(x, distance_matrix))

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

def swap_mutation(chromosome):
    idx1, idx2 = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def inversion_mutation(chromosome):
    start, end = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[start:end + 1] = reversed(chromosome[start:end + 1])
    return chromosome

def genetic_algorithm(distance_matrix, population_size, generations, crossover_rate, mutation_rate, tournament_size):
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)
    best_solution = min(population, key=lambda x: calculate_total_distance(x, distance_matrix))
    best_distances = [calculate_total_distance(best_solution, distance_matrix)]
    stationary_count = 0
    stationary_threshold = 50  # Adjust as needed

    for generation in range(generations):
        new_population = []

        new_population.append(best_solution)

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, distance_matrix, tournament_size)
            parent2 = tournament_selection(population, distance_matrix, tournament_size)

            if random.random() < crossover_rate:
                if random.random() < 0.5:
                    child = order_crossover(parent1, parent2)
                else:
                    child = pmx_crossover(parent1, parent2)
            else:
                child = parent1

            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    child = swap_mutation(child)
                else:
                    child = inversion_mutation(child)

            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda x: calculate_total_distance(x, distance_matrix))

        if current_best == best_solution:
            stationary_count += 1
        else:
            stationary_count = 0

        if calculate_total_distance(current_best, distance_matrix) < calculate_total_distance(best_solution, distance_matrix):
            best_solution = current_best

        best_distances.append(calculate_total_distance(best_solution, distance_matrix))

        if stationary_count >= stationary_threshold:
            print(f"Reached a stationary state after {generation} generations.")
            break

    return best_solution, best_distances

def plot_evolution(best_distances):
    plt.plot(best_distances)
    plt.xlabel('Generation')
    plt.ylabel('Total Distance')
    plt.title('Evolution of Total Distance in Genetic Algorithm')
    plt.show()

def run_algorithm_with_parameters(distance_matrix, population_size, generations, crossover_rate, mutation_rate, tournament_size):
    best_solution, best_distances = genetic_algorithm(distance_matrix, population_size, generations, crossover_rate, mutation_rate, tournament_size)
    
    print(f"Best solution: {best_solution}")
    print(f"Total distance: {calculate_total_distance(best_solution, distance_matrix)}")

    plot_evolution(best_distances)

if __name__ == "__main__":
    file_path = 'dataset/5.txt'
    distance_matrix = load_distance_matrix(file_path)

    population_size = 100
    generations = 1000
    crossover_rate = 0.8
    mutation_rate = 0.1
    tournament_size = 5

    run_algorithm_with_parameters(distance_matrix, population_size, generations, crossover_rate, mutation_rate, tournament_size)
