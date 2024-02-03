# This is not useful anymore, we have made a separate 5 files for 5 dataset, and we rename the dataset files too

# main.py
import random  # Add this line
import matplotlib.pyplot as plt
from tsp_utils import load_tsp_data, initialize_population, calculate_total_distance
from genetic_algorithm import tournament_selection, roulette_wheel_selection, order_crossover, partially_matched_crossover, swap_mutation, inversion_mutation
# from genetic_algorithm import genetic_algorithm

def genetic_algorithm(file_path, population_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.4):
    problem = load_tsp_data(file_path)
    num_cities = problem.dimension
    
    population = initialize_population(population_size, num_cities)

    best_distances = []

    for generation in range(generations):
        selected_parents = tournament_selection(population, k=5, problem=problem)
        offspring = []

        for i in range(0, len(selected_parents), 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = inversion_mutation(child2)

            offspring.extend([child1, child2])

        population = offspring

        best_solution = min(population, key=lambda x: calculate_total_distance(x, problem))
        best_distance = calculate_total_distance(best_solution, problem)
        best_distances.append(best_distance)

        # Print progress
        print(f"Genetic Algorithm Progress: {100 * generation / generations:.2f}%")

    return best_solution, best_distance, best_distances

# Example usage:
# file_path = "C:\\Users\\PRJA\\OneDrive - Carlson Rezidor\\Desktop\\A4\\att48.tsp"
# file_path = "C:\\Users\\PRJA\\OneDrive - Carlson Rezidor\\Desktop\\A4\\dantzig42.tsp"
# best_solution, best_distance, best_distances = genetic_algorithm(file_path)



def main():
    file_path = "C:\\Users\\PRJA\\OneDrive - Carlson Rezidor\\Desktop\\A4\\att48.tsp"
    best_solution, best_distance, best_distances = genetic_algorithm(file_path)

    # Print results and plot
    print("Best Solution:", best_solution)
    print("Best Distance:", best_distance)
    print("\nGenetic Algorithm Parameters:")
    print("Population Size: 100")
    print("Generations: 500")
    print("Crossover Rate: 0.8")
    print("Mutation Rate: 0.2")

    # Plot the evolution of the minimum total traveling distance
    plt.plot(range(len(best_distances)), best_distances)
    plt.xlabel("Generation")
    plt.ylabel("Total Distance")
    plt.title("Evolution of Minimum Total Traveling Distance")
    plt.show()
   
if __name__ == "__main__":
    main()
