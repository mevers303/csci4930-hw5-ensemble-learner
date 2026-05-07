#!/usr/bin/env python3

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# hyperparameters
input_file = "./dataset/dataset.csv"
num_rounds = 100
n_classes = 7



# the transformer we'll use
numeric_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                    'Horizontal_Distance_To_Fire_Points']




def load_data(input_file, transformer):    
    # load the data
    print("Loading data...")
    df = pd.read_csv(input_file, index_col=0)
    X = df.drop("Cover_Type", axis=1)
    Y = df["Cover_Type"]

    # scale the numeric features
    print("Scaling numeric features...")
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1234)
    # transform the training and testing data
    transformer.fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test, y_train, y_test, transformer



def training_adaboost(X_train, y_train, num_rounds):
    global weak_learners
    global alphas

    print("Training AdaBoost with SAMME...")

    # initialize the weights for the training examples
    weights = np.array([1/len(X_train)] * len(X_train))

    # for plotting the training and testing accuracy over time
    train_accuracies = []

    # train the weak learners
    for i in range(num_rounds):
        # train the current weak learner
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train, sample_weight=weights)

        # calculate the error and alpha for the current weak learner
        y_pred = model.predict(X_train)
        # instead of looping, let's use numpy functions.  y_pred != y_train will be zeros where the prediction is correct and ones where it's incorrect.  Then we can multiply by the weights and sum to get the total error.
        error = np.sum(weights * (y_pred != y_train)) / np.sum(weights)
        alpha = np.log((1 - error) / error) + np.log(n_classes - 1)
        # update the weights for the next round
        weights *= np.exp(alpha * (y_pred != y_train))
        weights /= np.sum(weights)

        # save the current weak learner and its alpha
        weak_learners.append(model)
        alphas.append(alpha)

        # calculate the training accuracy for this round
        train_accuracies.append(accuracy_score(y_train, y_pred))

    return weak_learners, alphas, train_accuracies



def testing_adaboost(X_test, y_test, weak_learners, alphas):
    global n_classes

    print("Testing AdaBoost...")

    n_samples = len(X_test)
    test_accuracies = []

    # matrix for results of each model
    final_predictions = np.zeros((n_samples, n_classes))
    
    # loop through weak learners to build the matrix
    for alpha, model in zip(alphas, weak_learners):
        y_pred = model.predict(X_test)
        # numpy to the rescue again
        rows_i = np.arange(n_samples)
        final_predictions[rows_i, y_pred - 1] += alpha
        # also calculate the accuracy for this round
        if y_test is not None:
            test_accuracies.append(accuracy_score(y_test, y_pred))

    # get the final predicted class for each test example
    final_predictions = np.argmax(final_predictions, axis=1) + 1

    final_accuracy = accuracy_score(y_test, final_predictions)
    return final_accuracy, test_accuracies


def base_model(X_train, y_train, X_test, y_test):
    # fit model
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)

    # predict on training and testing data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate metrics for training and testing data
    base_train_accuracy = accuracy_score(y_train, y_pred_train)
    base_test_accuracy = accuracy_score(y_test, y_pred_test)

    return base_train_accuracy, base_test_accuracy


def plot_metrics(train_accuracies, test_accuracies, base_train_accuracy, base_test_accuracy, final_accuracy):
    global num_rounds

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), train_accuracies, label='AdaBoost Train Acc')
    plt.plot(range(1, num_rounds + 1), test_accuracies, label='AdaBoost Test Acc')
    plt.axhline(y=base_test_accuracy, color='r', linestyle='--', label='Baseline (Single LR) Test Acc')
    plt.axhline(y=base_train_accuracy, color='g', linestyle='--', label='Baseline (Single LR) Train Acc')
    plt.axhline(y=final_accuracy, color='b', linestyle='--', label='AdaBoost Final Acc')
    plt.xlabel('Number of Rounds')
    plt.ylabel('Accuracy')
    plt.title('AdaBoost Accuracy vs. Number of Rounds')
    plt.legend()
    plt.savefig('results.png')
    plt.show()



def main():
    # the transformer we'll use
    transformer = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features)
    ], remainder="passthrough")

    # load the data
    X_train, X_test, y_train, y_test, transformer = load_data(input_file, transformer)

    # a list of the weak learners and their corresponding alphasl
    weak_learners = []
    alphas = []
    # metrics is going to be a list of tuples: ("accuracy", "precision", "recall", "f1_score")

    # Task 1: train the model
    weak_learners, alphas, train_accuracies = training_adaboost(X_train, y_train, num_rounds)
    # test the model
    final_accuracy, test_accuracies = testing_adaboost(X_test, y_test, weak_learners, alphas)

    # Task 2: train and test the base model
    base_train_accuracy, base_test_accuracy = base_model(X_train, y_train, X_test, y_test)

    # Task 3: plot the training and testing accuracy over time
    plot_metrics(train_accuracies, test_accuracies, base_train_accuracy, base_test_accuracy, final_accuracy)
