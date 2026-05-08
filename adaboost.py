#!/usr/bin/env python3

# Mark Evers
# 5/7/2026
# CSCI 4930 - Machine Learning
# Homework 5 - AdaBoost

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
judge_file = "./dataset/judge-no-labels.csv"
num_rounds = 100
n_classes = 7
lr_max_iter = 1000



# columns we need to scale
numeric_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                    'Horizontal_Distance_To_Fire_Points']




def load_data(input_file, transformer):    
    print("Loading data...")

    # load data from the csv
    df = pd.read_csv(input_file, index_col=0)
    X = df.drop("Cover_Type", axis=1)
    Y = df["Cover_Type"]
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1234)

    print("Scaling numeric features...")
    # fit the transformer on the training dataand then transform the training and testing data
    transformer.fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    return X_train, X_test, y_train, y_test, transformer



def training_adaboost(X_train, y_train, num_rounds, weak_learners=[], alphas=[]):
    print("Training AdaBoost with SAMME...")

    # initialize the weights for the training examples
    weights = np.array([1/len(X_train)] * len(X_train))

    # for plotting the training and testing accuracy over time
    individual_train_accuracies = []

    # train the weak learners
    for i in range(num_rounds):
        # train the current weak learner and get its predictions on the training data
        model = LogisticRegression(solver="lbfgs", max_iter=lr_max_iter, C=np.inf)
        model.fit(X_train, y_train, sample_weight=weights)
        y_pred = model.predict(X_train)

        # calculate the error and alpha for the current weak learner
        # instead of looping, let's use a numpy mask
        error = np.sum(weights * (y_pred != y_train)) / np.sum(weights)
        alpha = np.log((1 - error) / error) + np.log(n_classes - 1)
        # update the weights for the next round
        weights *= np.exp(alpha * (y_pred != y_train))
        weights /= np.sum(weights)

        # save the current weak learner and its alpha
        weak_learners.append(model)
        alphas.append(alpha)

        # calculate the training accuracy for this round
        individual_train_accuracies.append(accuracy_score(y_train, y_pred))

    return weak_learners, alphas, individual_train_accuracies



def testing_adaboost(weak_learners, alphas, X, y_true=None):
    global n_classes
    print("Testing AdaBoost...")

    # get the number of samples
    n_samples = len(X)

    # matrix for results of each model
    weak_learner_predictions = np.zeros((n_samples, n_classes))
    # list of model accuracies for each round
    ensemble_test_accuracies = []
    individual_test_accuracies = []
    
    # loop through weak learners to build the matrix
    for alpha, model in zip(alphas, weak_learners):
        y_pred = model.predict(X)
        # numpy to the rescue again
        rows_i = np.arange(n_samples)
        weak_learner_predictions[rows_i, y_pred - 1] += alpha

        # also calculate the accuracy for this round
        if y_true is not None:
            individual_test_accuracies.append(accuracy_score(y_true, y_pred))
        
        # get the ensemble predictions for this round and calculate the accuracy
        ensemble_predictions = np.argmax(weak_learner_predictions, axis=1) + 1
        if y_true is not None:
            ensemble_test_accuracies.append(accuracy_score(y_true, ensemble_predictions))

    # if we have the true labels, calculate the final accuracy, otherwise just return the predictions
    if y_true is not None:
        return ensemble_test_accuracies, individual_test_accuracies
    else:
        # this is bad practice to have a function return two different types of output, but it will work for our purposes
        return ensemble_predictions



def base_model_predictions(X_train, y_train, X_test, y_test):
    print("Training and testing baseline model...")

    # fit model
    model = LogisticRegression(solver="lbfgs", max_iter=lr_max_iter)
    model.fit(X_train, y_train)

    # predict on training and testing data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate metrics for training and testing data
    base_train_accuracy = accuracy_score(y_train, y_pred_train)
    base_test_accuracy = accuracy_score(y_test, y_pred_test)

    return base_train_accuracy, base_test_accuracy


def plot_metrics(individual_train_accuracies, individual_test_accuracies, base_train_accuracy, base_test_accuracy, ensemble_test_accuracies):
    global num_rounds
    print("Plotting metrics...")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_rounds + 1), individual_train_accuracies, label='Weak Learners Train Accuracy')
    plt.plot(range(1, num_rounds + 1), individual_test_accuracies,  label='Weak Learners Test Accuracy')
    plt.plot(range(1, num_rounds + 1), ensemble_test_accuracies, label='Ensemble Test Accuracy')
    plt.axhline(y=base_test_accuracy, color='r', linestyle='--',  label='Baseline (Logistic Regression) Test Accuracy')
    plt.axhline(y=base_train_accuracy, color='g', linestyle='--', label='Baseline (Logistic Regression) Train Accuracy')
    plt.xlabel('Round Number')
    plt.ylabel('Accuracy')
    plt.title('AdaBoost Accuracy vs. Number of Rounds')
    plt.legend()
    plt.savefig('results.png')
    plt.show()



def predict_judge_data(transformer, weak_learners, alphas):
    global judge_file
    print("Predicting on judge dataset...")

    # load data from the csv
    df = pd.read_csv(judge_file, index_col=0)
    # transform the data
    X = transformer.transform(df)

    # get predictions from the model
    y_pred = testing_adaboost(weak_learners, alphas, X)

    # save the predictions to a csv
    output_df = pd.DataFrame({"Id": df.index, "Cover_Type": y_pred})
    output_df.to_csv("judge_predictions.csv")

    return output_df




def main():
    # the transformer we'll use
    transformer = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features)
    ], remainder="passthrough")

    # load the data
    X_train, X_test, y_train, y_test, transformer = load_data(input_file, transformer)

    # Task 1: train the model
    weak_learners, alphas, individual_train_accuracies = training_adaboost(X_train, y_train, num_rounds)
    # test the model
    ensemble_test_accuracies, individual_test_accuracies = testing_adaboost(weak_learners, alphas, X_test, y_test)

    # Task 2: train and test the base model
    base_train_accuracy, base_test_accuracy = base_model_predictions(X_train, y_train, X_test, y_test)

    # Task 3: plot the training and testing accuracy over time
    plot_metrics(individual_train_accuracies, individual_test_accuracies, base_train_accuracy, base_test_accuracy, ensemble_test_accuracies)

    # Task 4
    # After analyzing the plot, the weak learners become much less accurate after the first round.  This is expected behavior as the model focuses on the predictions that it got wrong by weighting those samples higher as time goes on.
    # The ensemble model starts off at the same accuracy as the first weak learner, and then declines and plateaus as the weak learners become less accurate due to overfitting.  The outliers in the training data are likely causing the weak learners to become less accurate as they focus more and more on those outliers, which may not be representative of the overall data distribution.  The baseline model performs better than the ensemble model after a certain number of rounds, which suggests that the ensemble is overfitting to the training data and not generalizing well to the test data.
    # I would like to see what happens if I were to use decision trees as the weak learners instead of logistic regression, as decision trees are more commonly used as weak learners in AdaBoost and may be less prone to overfitting in this case.

    # Task 5: Predict on the judge dataset
    predict_judge_data(transformer, weak_learners, alphas)

    print("Done!")




if __name__ == "__main__":
    main()
