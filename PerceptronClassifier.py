import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.all_classes = set()
        self.multi_weight_vecs = []
        self.max_iteration = 0
        self.num_of_features = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        x_with_bias = []
        for item in X:
            x_with_bias.append(np.insert(item, 0, 1))
        self.multi_weight_vecs = []
        find_wrong_sample = 1
        self.max_iteration = len(X)
        # num of column
        self.num_of_features = len(x_with_bias[1])
        self.all_classes = set(y)
        for i in self.all_classes:
            w = [0] * (self.num_of_features)
            self.multi_weight_vecs.append(w)
        while find_wrong_sample == 1:
            # creating empty vec for each class (wight vec)
            for j in range(self.max_iteration):
                predict_class = list(self.all_classes)[0]
                max_arg = 0
                # decide how to classify the new sample according to multi class rule
                for class1 in self.all_classes:
                    find_dot_product = np.dot(x_with_bias[j], self.multi_weight_vecs[class1])
                    if find_dot_product >= max_arg:
                        max_arg = find_dot_product
                        predict_class = class1

                # if the predicted class is wrong we need to update2 vecs
                # the first is the wight vec of the real class
                # the second is the wight vec ot the wrong class
                # if the prediction is the correct class we do nothing
                if y[j] != predict_class:
                    self.multi_weight_vecs[y[j]] += x_with_bias[j]
                    self.multi_weight_vecs[predict_class] -= x_with_bias[j]
                    find_wrong_sample = 1
                    break
                else:
                    find_wrong_sample = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        x_with_bias = []
        for item in X:
            x_with_bias.append(np.insert(item, 0, 1))
        x_with_bias1 = np.array(x_with_bias)
        predict_list = np.zeros(np.shape(X)[0], dtype=np.uint8)
        self.max_iteration = len(x_with_bias)
        for j in range(self.max_iteration):
            predict_class = list(self.all_classes)[0]
            max_arg = 0
            for class1 in self.all_classes:
                find_dot_product = np.dot(x_with_bias1[j], self.multi_weight_vecs[class1])
                if find_dot_product > max_arg:
                    max_arg = find_dot_product
                    predict_class = class1
            predict_list[j] = predict_class
        return predict_list

        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


if __name__ == "__main__":
    print("*" * 20)
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
