import numpy as np


class Perceptron(object):

    def __init__(self, threshold=100, step_size=1.0):
        self.step_size = step_size
        self.weights = np.array([0, 0, 0, 0])
        self.data = None
        self.labels = None
        self.threshold = threshold
        self.bias = 0.0

    def sign(self, n):
        if n >= 0.0:  # if incorrectly classified
            return 1.0
        else:
            return 0.0

    def prediction(self, inputs, label, counter):
        if counter == 0:
            return 1.0

        sum = np.dot(inputs, self.weights) + self.bias # w^Tx + b
        sum = sum * label * -1  # -y * w^Tx + b
        output = self.sign(sum)
        return output



    def train(self):
        counter = 0
        labels = self.labels
        training_input = self.data

        for _ in range(self.threshold):
            weights_mult = np.array([0, 0, 0, 0])
            bias_mult = 0.0
            for inputs, label in zip(training_input, labels):
                predict = self.prediction(inputs, label, counter)
                if predict == 1.0:
                    weights_mult = np.add(weights_mult, float(label) * inputs)
                    bias_mult += label
                # else:
                #     weights_mult = np.array([0, 0, 0, 0])
                #     bias_mult = 0.0

            self.weights = np.add(self.weights, self.step_size * weights_mult)
            self.bias = self.bias + self.step_size * bias_mult

            ls = self.weights.tolist()
            if counter < 3:
                print("Weights: " + str(ls))
                print("Bias: " + str(self.bias))
                print("Iteration: " + str(counter) + '\n')
            if self.has_converged():
                print("Final weights, biases, and iteration")
                print("Weights: " + str(ls))
                print("Bias: " + str(self.bias))
                print("Iteration: " + str(counter) + '\n')
                return
            counter += 1

    def has_converged(self):
        for expected, data_line in zip(self.labels, self.data):
            guess = self.test_classify(data_line, expected)
            if 1.0 == guess:
                return 0
        return 1

    def test_classify(self, inputs, label):
        sum = np.dot(inputs, self.weights) + self.bias
        sum = sum * label * -1
        if sum >= 0:
            return 1.0
        elif sum < 0:
            return -1.0

    def read_data(self, filename):
        data = []
        with open(filename) as file:
            for line in file:
                data_line = [float(item) for item in line.split(',')]
                data.append(data_line)
        self.data = np.array(data)
        self.labels = self.data[:, -1]  # grab all the labels
        self.data = self.data[:, 0:4]  # grab all the data points


class Stochastic(object):

    def __init__(self, threshold=1000000000000000, step_size=1.0):
        self.step_size = step_size
        self.weights = np.array([0, 0, 0, 0])
        self.data = None
        self.labels = None
        self.threshold = threshold
        self.bias = 0.0

    def sign(self, n):
        if n >= 0.0:
            return 1.0
        else:
            return 0.0

    def prediction(self, inputs, label, counter):
        if counter == 0:
            return 1.0

        sum = np.dot(inputs, self.weights) + self.bias
        sum = sum * label * -1
        output = self.sign(sum)
        return output

    def test_classify(self, inputs, label):
        sum = np.dot(inputs, self.weights) + self.bias
        sum = sum * label * -1
        if sum >= 0:
            return 1.0
        elif sum < 0:
            return -1.0

    def train(self):
        counter = 0
        position_in_dataset = 0

        for _ in range(self.threshold):
            weights_mult = np.array([0, 0, 0, 0])
            bias_mult = 0.0
            gradient_has_changed = False
            predict = self.prediction(self.data[position_in_dataset], self.labels[position_in_dataset], counter)

            if predict == 1.0:
                weights_mult = np.add(weights_mult, float(self.labels[position_in_dataset]) * self.data[position_in_dataset])
                bias_mult = self.labels[position_in_dataset]
                gradient_has_changed = True

            self.weights = np.add(self.weights, self.step_size * weights_mult)
            self.bias = self.bias + self.step_size * bias_mult

            ls = self.weights.tolist()
            if counter < 3:
                print("Weights: " + str(ls))
                print("Bias: " + str(self.bias))
                print("Iteration: " + str(counter) + '\n')
            if gradient_has_changed:
                if self.has_converged():
                    print("Final weights, biases, and iteration")
                    print("Weights: " + str(ls))
                    print("Bias: " + str(self.bias))
                    print("Iteration: " + str(counter) + '\n')
                    return
            counter += 1
            position_in_dataset = (position_in_dataset + 1) % len(self.data)
            

    def has_converged(self):
        for expected, data_line in zip(self.labels, self.data):
            guess = self.test_classify(data_line, expected)
            if 1.0 == guess:
                return 0
        return 1

    # returns data in array from given filename
    def read_data(self, filename):
        data = []
        with open(filename) as file:
            for line in file:
                data_line = [float(item) for item in line.split(',')]
                data.append(data_line)
        self.data = np.array(data)
        self.labels = self.data[:, -1] # grab all the labels
        self.data = self.data[:, 0:4] # grab all the vectors
        # print(self.data)


# MAIN
x = 0.001
print("<--- Problem 1a. --->")
p = Perceptron(threshold=1000, step_size=x)
p.read_data("perceptron.data")
p.train()

print("step size = " + str(x))

print("--- Problem 1b. ---")
s = Stochastic(step_size=x)
s.read_data("perceptron.data")
s.train()
print("step size = " + str(x))
