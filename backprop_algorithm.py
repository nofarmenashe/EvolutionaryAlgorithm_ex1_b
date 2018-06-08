import random
import numpy as np


# activation functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    return (1 - sigmoid(z)) * sigmoid(z)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
   return 1 - np.power(np.tanh(z), 2)


def RELU(z):
    return np.maximum(z, 0, z)


def d_RELU(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class BackpropArgs:
    def __init__(self, input_size, output_size, learning_rate=0.01, hidden_layers_sizes=[], epochs=5, f=RELU, df=d_RELU):

        self.learning_rate = learning_rate
        self.hidden_layers_sizes = hidden_layers_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.f = f
        self.df = df

        self.epochs = epochs

    def create_layers_list(self):
        layers_list = [self.input_size]
        layers_list.extend(self.hidden_layers_sizes)
        layers_list.append(self.output_size)
        return layers_list

    def choose_hyper_params(self):
        self.hidden_layers_sizes = self.choose_hidden_layers()
        self.f, self.df = self.choose_activation()
        self.learning_rate = self.choose_lr()
        self.epochs = self.choose_epochs()

    def choose_hidden_layers(self):
        layers = sorted(random.sample([64, 128, 256, 512, 768, 1024], k=2))
        layers = layers[::-1]
        return layers
        # return -np.sort(-np.array(random.sample(range(self.output_size, self.input_size), k=2)))

    def choose_activation(self):
        activation_functions = [(sigmoid, d_sigmoid), (tanh, d_tanh), (RELU, d_RELU)]
        return random.choice(activation_functions)

    def choose_lr(self):
        return random.uniform(0.001, 0.02)

    def choose_epochs(self):
        return random.randint(1, 20)


class BackPropModel:

    def __init__(self, args: BackpropArgs):
        self.num_layers = len(args.hidden_layers_sizes) + 2

        # initialising network parameters
        self.args = args
        self.layers = self.args.create_layers_list()
        self.biases = np.array([np.zeros((y, 1)) for y in self.layers[1:]])
        self.weights = np.array([np.random.normal(loc=0.0, scale=0.1, size=(y, x))
                                 for x, y in list(zip(self.layers[:-1], self.layers[1:]))])

    def forward(self, x):
        x = np.reshape(x, (len(x), 1))
        for w, b in zip(self.weights, self.biases):
            x = self.args.f(w.dot(x)) + b

        return softmax(x)

    def backprop(self, x, y):
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        v = x
        z_list = []
        v_list = [np.reshape(v, (len(v), 1))]
        for w, b in zip(self.weights, self.biases):
            z = w.dot(np.reshape(v, (len(v), 1))) + b
            z_list.append(z)
            v = self.args.f(z)
            v_list.append(v)
        v_list[-1] = softmax(v)

        # Backwords
        delta = (v_list[-1] - y)
        weight_gradients[-1] = delta.dot(v_list[-2].T)
        bias_gradients[-1] = delta

        for l in range(2, self.num_layers):
            z = z_list[-l]
            delta = self.args.df(z) * self.weights[-l + 1].T.dot(delta)
            weight_gradients[-l] = delta.dot(v_list[-l - 1].T)
            bias_gradients[-l] = delta

        return weight_gradients, bias_gradients

    def update_params(self, weight_gradients, bias_gradients):
        for i in range(len(self.weights)):
            self.weights[i] += -self.args.learning_rate * weight_gradients[i]
            self.biases[i] += -self.args.learning_rate * bias_gradients[i]

    def train(self, training_data, validation_data=None):
        for i in range(self.args.epochs):
            random.shuffle(training_data)
            inputs = [data[0] for data in training_data]
            targets = [data[1] for data in training_data]

            for j in range(len(inputs)):
                weight_gradients, bias_gradients = self.backprop(inputs[j], targets[j])
                self.update_params(weight_gradients, bias_gradients)

            # print("{} epoch(s) done".format(i + 1))
            if validation_data:
                print("Validation Accuracy:", str(self.test(validation_data)) + "%")
        # print("Training done.")

    def test(self, dataset):
        test_results = [(np.argmax(self.forward(x[0])), np.argmax(x[1])) for x in dataset]
        return float(sum([int(x == y) for (x, y) in test_results])) / len(dataset) * 100
