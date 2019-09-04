import numpy as np
import cvxopt


class SVM(object):

    def __init__(self):
        pass

    def read_data(self, filename):
        data = []
        with open(filename) as file:
            for line in file:
                data_line = [float(item) for item in line.split(',')]
                data.append(data_line)
        self.data = np.array(data)
        self.labels = self.data[:, -1] # grab all the labels
        self.data = self.data[:, 0:4] # grab all the vectors

