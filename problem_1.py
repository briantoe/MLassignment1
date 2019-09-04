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
        self.threshold = threshold
        self.bias = 0.0


    def sign(self, n):
        if n > 0.0:
            return 1.0
        else:
            return 0.0

    def prediction(self, inputs, counter):
        if counter == 0:
            return 1.0
        print(self.weights)
        print()
        sum = np.dot(inputs, self.weights) + self.bias
        sum = sum * inputs * -1
        output = self.sign(sum)
        return output

    def train(self):
        counter = 0
        labels = []
        for line in self.data:
            labels.append(line[-1])
        weights_mult = np.array([0, 0, 0, 0])
        bias_mult = 0.0


        training_input = []
        for line in self.data:
            training_input.append(line[:-1])
        training_input = np.array(training_input)

        for i in range(self.threshold):
            for inputs, label in zip(training_input, labels):

                predict = self.prediction(inputs, counter)

                if predict == 1.0:
                    # self.weights[:-1] = np.add(self.weights[:-1], self.step_size * (label - predict) * inputs[:-1])
                    weights_mult = np.add(weights_mult, float(label) * inputs)
                    bias_mult += float(label)

                # self.weights[-1] += self.step_size * (label - predict)

            self.weights = np.add(self.weights, self.step_size * weights_mult)
            self.bias += self.step_size * bias_mult

            if counter < 3:
                ls = self.weights.tolist()
                print("Weights: " + str(ls[:-1]))
                print("Bias: " + str(ls[-1]))
                print("Iteration: " + str(counter) + '\n')
            if self.has_converged():
                ls = self.weights.tolist()
                print("Weights: " + str(ls[:-1]))
                print("Bias: " + str(ls[-1]))
                print("Iteration: " + str(counter) + '\n')
                return
            counter += 1

        print("Weights: " + str(ls[:-1]))
        print("Bias: " + str(ls[-1]))
        print("Iteration: " + str(counter) + '\n')
        return

    def has_converged(self):
        # print("testing for convergence")
        for line in self.data:
            if line[-1] != self.prediction(line, -1):
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


p = Perceptron()
p.read_data("perceptron.data")
p.train()

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




