import operator
import random
import numpy as np

from backprop_algorithm import BackPropModel, BackpropArgs


def calculate_probability(p):
    return random.random() >= 1-p


class GAArgs:
    def __init__(self, population_size, replication_rate, mutation_rate, elitism_rate):
        self.population_size = population_size
        self.replication_rate = replication_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate


class GAModel:
    def __init__(self, args: GAArgs):
        self.population_size = args.population_size
        self.replication_rate = args.replication_rate
        self.mutation_rate = args.mutation_rate
        self.elitism_rate = args.elitism_rate
        self.population = self.init_population()

    def generate_network_args(self):
        babprop_args = BackpropArgs(28*28, 10)
        babprop_args.choose_hyper_params()
        # print(babprop_args.epochs, babprop_args.learning_rate, babprop_args.hidden_layers_sizes,
        #       babprop_args.f, babprop_args.df)
        return babprop_args

    def init_population(self):
        population = []
        for i in range(self.population_size):
            backprop_args = self.generate_network_args()
            population.append(BackPropModel(backprop_args))
        return population

    def fitness(self, nn_chromosome: BackPropModel, train_dataset, val_dataset):
        nn_chromosome.train(train_dataset)
        accuracy = nn_chromosome.test(val_dataset)
        # print(str(accuracy) + "%")
        return accuracy

    def replication(self, population_list):
        return random.sample(population_list, k=int(self.replication_rate * self.population_size))

    def choose_parents(self, population_fitness_tuples):
        networks, fitnesses = zip(*population_fitness_tuples)
        sum_fitnesses = np.sum(fitnesses)
        fitnesses = [float(fitness) / sum_fitnesses for fitness in fitnesses]
        return np.random.choice(networks, 2, p=fitnesses)

    def breed_layers(self, p1_layers, p2_layers):
        hidden_layers = []
        child_layers = []

        hidden_layers.extend(p1_layers)
        hidden_layers.extend(p2_layers)
        # print(hidden_layers)
        child_layers.extend(random.sample(hidden_layers, k=2))
        layers = -np.sort(-np.array(child_layers))
        # print("child", layers)
        return layers

    def breed_parents(self, p1: BackPropModel, p2: BackPropModel):
        # lr = random.sample([p1.args.learning_rate, p2.args.learning_rate], k=1)[0]
        # epochs = random.sample([p1.args.epochs, p2.args.epochs], k=1)[0]
        lr = 0.01
        epochs = 10
        activation, d_activation = random.sample([(p1.args.f, p1.args.df), (p2.args.f, p2.args.df)], k=1)[0]
        hidden_layers = self.breed_layers(list(p1.args.hidden_layers_sizes), list(p2.args.hidden_layers_sizes))

        child_args = BackpropArgs(p1.args.input_size, p1.args.output_size, lr, hidden_layers, epochs, activation, d_activation)

        return BackPropModel(child_args)

    def crossover(self, population_fitness_tuples, num_of_crossovers):
        children = []
        for i in range(num_of_crossovers):
            p1, p2 = self.choose_parents(population_fitness_tuples)
            children.append(self.breed_parents(p1, p2))
        return children

    def mutate(self, chromosome: BackPropModel):
        parameter_to_mutate = random.randint(1, 2)
        if parameter_to_mutate == 1:
            print("mutate hidden layers")
            chromosome.args.hidden_layers_sizes = chromosome.args.choose_hidden_layers()
            chromosome.layers = chromosome.args.create_layers_list()
        if parameter_to_mutate == 2:
            print("mutate f")
            chromosome.args.f, chromosome.args.df = chromosome.args.choose_activation()
        return chromosome

    def population_mutation(self, population):
        mutated_population = []
        for chromosome in population:
            if calculate_probability(self.mutation_rate):
                mutated_population.append(self.mutate(chromosome))
            else:
                mutated_population.append(chromosome)
        return mutated_population

    def train(self, train_dataset, val_dataset, test_dataset):
        best_fitness = (None, 0)
        population_fitnesses = []
        while best_fitness[1] < 98:
            new_population = []

            # train_batch = random.sample(train_dataset, k=100)
            # print("Shuffle Data")
            # random.shuffle(train_dataset)
            # random.shuffle(test_dataset)

            # calculate fitnesses
            print("Calc Fitnesses")
            # if population_fitnesses:
            #     # filter all pop fit of not changed
            population_fitnesses.extend([(nn, self.fitness(nn, train_dataset[:1000], val_dataset))
                                         for nn in self.population])

            population_fitnesses.sort(key=operator.itemgetter(1))
            population_fitnesses = population_fitnesses[::-1]

            best_fitness = population_fitnesses[0]
            print(best_fitness[1])

            num_of_elit = int(self.elitism_rate * self.population_size)

            # replication - select randomly from the rest
            print("Start Replication")
            rest_of_population = [population_fitness[0] for population_fitness
                                  in population_fitnesses[int(self.elitism_rate * self.population_size):]]
            new_population.extend(self.replication(rest_of_population))

            # crossover - breed random parents
            print("Start Crossover")
            num_of_chromosomes_left = self.population_size - len(new_population) - num_of_elit
            new_population.extend(self.crossover(population_fitnesses, num_of_chromosomes_left))

            # mutation - mutate new population
            print("Start Mutation")
            new_population = self.population_mutation(new_population)

            # elitism - select top
            print("Start Elitism")
            elit_chromosomes = [population_fitness[0] for population_fitness
                                   in population_fitnesses[:num_of_elit]]
            new_population.extend(elit_chromosomes)

            print("Finish Generation")
            self.population = new_population

        accuracy = best_fitness[0].test(test_dataset)
        print("Test Acuuracy: " + str(accuracy))
