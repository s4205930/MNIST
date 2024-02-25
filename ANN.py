import numpy as np
import pandas as pd

data = pd.read_csv('MNIST.csv')
data = np.array(data)
np.random.shuffle(data)

m, n = data.shape

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255

_, m_train = X_train.shape

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

class Neural_Network:
    def __init__(self, input_size, hidden_1_size, output_size, bias = 1):
        self.input_size = input_size
        self.hidden_1_size = hidden_1_size
        self.output_size = output_size

        self.W1 = np.random.rand(hidden_1_size, input_size) - 0.5
        self.b1 = np.random.rand(hidden_1_size, bias) - 0.5

        self.W2 = np.random.rand(output_size, hidden_1_size) - 0.5
        self.b2 = np.random.rand(output_size, bias) - 0.5



    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)

        Z2 = self.W2.dot(A1)
        A2 = softmax(Z2)

        return A2


pop_size = 20
mutation_rate = 0.05
generations = 100

def fitness_function(network, X, y):
    nn = Neural_Network(input_size=784, hidden_1_size=10, output_size=10)
    nn.W1 = network['W1']
    nn.b1 = network['b1']
    nn.W2 = network['W2']
    nn.b2 = network['b2']

    predictions = nn.forward_prop(X)
    accuracy = np.mean(np.argmax(predictions, axis=0) == y)
    return accuracy

def init_pop(nn, pop_size):
    population = []

    for _ in range(pop_size):
        network = {
            'W1': nn.W1.copy(),
            'b1': nn.b1.copy(),
            'W2': nn.W2.copy(),
            'b2': nn.b2.copy(),
        }
        population.append(network)
    return population

def select_parents(population, X, y):
    fitness_scores = [fitness_function(network, X, y) for network in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    selected_indices = np.random.choice(len(population), size = 2, p = probabilities, replace = False)
    parent_1 = population[selected_indices[0]].copy()
    parent_2 = population[selected_indices[1]].copy()
    return parent_1, parent_2

def crossover(parent_1, parent_2):
    child = {}
    for key in parent_1:
        if np.random.rand() < 0.5:
            child[key] = parent_1[key]
        else:
            child[key] = parent_2[key]
    return child

def mutate(network, mutation_rate):
    for key in network:
        if np.random.rand() < mutation_rate:
            network[key] += np.random.randn(*network[key].shape) * 0.05
    return network

nn = Neural_Network(784, 10, 10)
population = init_pop(nn, pop_size)

for generation in range(generations):
    print(generation)
    new_population = []

    for _ in range(pop_size // 2):
        parent_1, parent_2 = select_parents(population, X_train, Y_train)

        child_1 = crossover(parent_1, parent_2)
        child_1 = mutate(child_1, mutation_rate)

        child_2 = crossover(parent_1, parent_2)
        child_2 = mutate(child_2, mutation_rate)

        new_population.extend([child_1, child_2])
    population = new_population

best_network = max(population, key=lambda x: fitness_function(x, X_train, Y_train))
print(round(fitness_function(best_network, X_test, Y_test) * 100, 4), '%')