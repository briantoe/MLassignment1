import numpy as np

# step_size = 1
# iters = 0
# classifier = None
# input = None
# perceptron_loss = 0


class Perceptron(object):

    def __init__(self, threshold=100, step_size=1.0):
        self.step_size = step_size
        self.weights = np.array([0, 0, 0, 0])
        self.data = None
        self.labels = None
        self.threshold = threshold
        self.bias = 0.0

    def sign(self, n):
        if n > 0.0:
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
        if sum > 0:
            return 1.0
        elif sum < 0:
            return -1.0

    def train(self):
        counter = 0
        labels = self.labels
        training_input = self.data

        for i in range(self.threshold):
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
            if True:
                print("Weights: " + str(ls))
                print("Bias: " + str(self.bias))
                print("Iteration: " + str(counter) + '\n')
            if self.has_converged():
                print("Weights: " + str(ls))
                print("Bias: " + str(self.bias))
                print("Iteration: " + str(counter) + '\n')
                return
            counter += 1

        ls = self.weights.tolist()
        print("Weights: " + str(ls))
        print("Bias: " + str(self.bias))
        print("Iteration: " + str(counter) + '\n')
        return

    def has_converged(self):
        for expected, data_line in zip(self.labels, self.data):
            guess = self.test_classify(data_line, expected)
            if  1.0 == guess:
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


p = Perceptron()
p.read_data("perceptron.data")
p.train()
print(p.has_converged())

# weights = np.array([0, 0, 0, 0, 0])
# weights = np.transpose(weights)
# bias = 0
# classifier = int(input[-1])
# del input[-1]
# input = np.array(input)
#
# weights = weights + step_size
#
# bias = bias + step_size




