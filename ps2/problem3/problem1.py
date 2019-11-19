import numpy as np


class Question:

    def __init__(self, col, val):
        self.col = col
        self.val = val

    def match(self, inputt): # puts given input into the question and evaulates boolean value
        val = inputt[self.col]
        return val == self.val

class Leaf:
    def __init__(self, rows):
        self.prediction = class_counts(rows)

class DecisionNode:

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            data_line = [item.replace('\n', '') for item in line.split(',')]
            data.append(data_line)
    data = np.array(data)
    return data
    # labels = data[:, 0] # grab labels
    # dim = len(data[0]) # dimension of the data
    # data = data[:, 1:dim] # grab vectors
    # return (data, labels)

def unique_values(rows, cols):
    return set([row[cols] for row in rows])


def find_best_split(rows):
    best_gain = 0
    best_question = None
    uncertain = gini(rows)
    feat_num = len(rows[0] - 1)

    for col in range(feat_num):
        values = unique_values(rows, col)

        for val in values:
            question = Question(col,val)
            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0: # does not divide dataset
                continue

            gain = information_gain(true_rows, false_rows, uncertain)

            if(gain > best_gain):
                best_gain = gain
                best_question = question

    return best_gain, best_question            

def class_counts(rows):
    counts = {}
    for item in rows: 
        label = item[0] # grab the label off of the datapoint
        if label not in counts: # this counts the amount of data per classifier, stores in a dict 
                                # to reference  for each classifier
            counts[label] = 0
        counts[label] += 1
    return counts

def information_gain(left,right, cur_uncertain):
    p = float(len(left))/ (len(left) + len(right))
    return cur_uncertain -  p * gini(left) - (1-p) * gini(right)

def gini(row): # rows
    
    counts = class_counts(row)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(row))
        impurity -= prob_of_lbl**2
    
    return impurity

def partition(rows, question):
    true_rows, false_rows = [], []

    for row in rows:
        if question.match() == True:
            true_rows.append(row)
        else:
            false_rows.append(row)

    return true_rows, false_rows      

def build_tree(rows):

    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return DecisionNode(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(row, Leaf):
        return node.prediction

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def main():

    # attribute_types = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    
    filename = "mush_train.data"
    labels_and_data = read_data(filename)

    filename = "mush_test.data"
    test_labels_and_data = read_data(filename)

    tree = build_tree(labels_and_data[0])
    print(classify(labels_and_data[0], tree))




if __name__ == "__main__":
    main()
