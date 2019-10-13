import numpy as np
from anytree import Node, RenderTree

def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            data_line = [item.replace('\n', '') for item in line.split(',')]
            data.append(data_line)
    # data = np.array(data)
    return data

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

def class_counts(rows):
    counts = {}
    for item in rows: 
        label = item[0] # grab the label off of the datapoint
        if label not in counts: # this counts the amount of data per classifier, stores in a dict 
                                # to reference  for each classifier
            counts[label] = 0
        counts[label] += 1
    return counts


def main():

    # attribute_types = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    
    filename = "mush_train.data"
    labels_and_data = read_data(filename)

    
    # print(str(labels_and_data))

    filename = "mush_test.data"
    test_labels_and_data = read_data(filename)


    # tree = build_tree(labels_and_data[0])
    # print(classify(labels_and_data[0], tree))




if __name__ == "__main__":
    main()
