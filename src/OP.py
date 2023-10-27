import random
import math

# Genetic Algorithm implementation
class GeneticAlgorithm:
    def __init__(self, objective_function, population_size, mutation_rate, crossover_rate):
        self.objective_function = objective_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self):
        # Initialize a random population
        population = []
        for _ in range(self.population_size):
            chromosome = [random.randint(0, 1) for _ in range(10)]  # Example: Binary chromosome of length 10
            population.append(chromosome)
        
        return population

    def evaluate_population(self, population):
        # Calculate the fitness of each chromosome in the population
        fitness_values = []
        for chromosome in population:
            fitness = self.objective_function(chromosome)
            fitness_values.append(fitness)
        
        return fitness_values

    def select_parents(self, population, fitness_values):
        # Select parents based on fitness values (e.g., using tournament selection)
        parents = []
        for _ in range(self.population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if fitness_values[population.index(parent1)] > fitness_values[population.index(parent2)]:
                parents.append(parent1)
            else:
                parents.append(parent2)

        return parents

    def crossover(self, parents):
        # Perform crossover between parents to create offspring
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            crossover_point = random.randint(1, len(parent1) - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            offspring.append(offspring1)
            offspring.append(offspring2)
        
        return offspring

    def mutate(self, population):
        # Mutate the population based on the mutation rate
        for i in range(self.population_size):
            for j in range(len(population[i])):
                if random.random() < self.mutation_rate:
                    population[i][j] = 1 - population[i][j]  # Flip the bit

        return population

    def run(self, max_generations):
        # Run the genetic algorithm
        population = self.initialize_population()
        best_solution = None

        for _ in range(max_generations):
            fitness_values = self.evaluate_population(population)
            
            # Find the best solution in the current generation
            best_index = fitness_values.index(max(fitness_values))
            best_solution = population[best_index]

            parents = self.select_parents(population, fitness_values)
            offspring = self.crossover(parents)
            population = self.mutate(offspring)

        return best_solution


# Simulated Annealing implementation
class SimulatedAnnealing:
    def __init__(self, objective_function, initial_solution, initial_temperature, cooling_rate):
        self.objective_function = objective_function
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def accept_move(self, new_solution):
        # Decide whether to accept a new solution based on the Metropolis criterion
        current_fitness = self.objective_function(self.current_solution)
        new_fitness = self.objective_function(new_solution)
        
        if new_fitness > current_fitness:
            return True
        else:
            acceptance_probability = math.exp((new_fitness - current_fitness) / self.temperature)
            return random.random() < acceptance_probability

    def generate_neighbor(self):
        # Generate a neighboring solution (e.g., by flipping a bit)
        neighbor = list(self.current_solution)
        index = random.randint(0, len(neighbor) - 1)
        neighbor[index] = 1 - neighbor[index]  # Flip the bit
        
        return neighbor

    def run(self, max_iterations):
        # Run the simulated annealing algorithm
        iteration = 0

        while iteration < max_iterations:
            new_solution = self.generate_neighbor()
            
            if self.accept_move(new_solution):
                self.current_solution = new_solution
            
            if self.objective_function(new_solution) > self.objective_function(self.best_solution):
                self.best_solution = new_solution

            self.temperature *= self.cooling_rate
            iteration += 1

        return self.best_solution
