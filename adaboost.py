#!/usr/bin/env python3

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, aucdi
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# hyperparameters
input_file = "./dataset/dataset.csv"
num_rounds = 100
n_classes = 7





# load the data
print("Loading data...")
df = pd.read_csv(input_file, index_col=0)
X = df.drop("Cover_Type", axis=1)
Y = df["Cover_Type"]


# scale the numeric features
print("Scaling numeric features...")
numeric_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                    'Horizontal_Distance_To_Fire_Points']
transformer = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features)
], remainder="passthrough")
transformer.fit(X)
X = transformer.transform(X)



# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1234)



# a list of the weak learners and their corresponding alphasl
weak_learners = []
alphas = []
weights = np.array([1/len(X_train)] * len(X_train))
# metrics is going to be a list of tuples: ("accuracy", "precision", "recall", "f1_score")
metrics = ["accuracy", "precision", "recall", "f1_score"]


def training_adaboost(X_train, y_train, num_rounds):
    global weights
    global weak_learners
    global alphas

    print("Training AdaBoost with SAMME...")

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


def testing_adaboost(X_test, y_test=None):
    global weak_learners
    global alphas

    print("Testing AdaBoost...")
    n_samples = len(X_test)

    # matrix for results of each model
    final_predictions = np.zeros((n_samples, n_classes))
    
    for alpha, model in zip(alphas, weak_learners):
        y_pred = model.predict(X_test)
        # numpy to the rescue again
        rows_i = np.arange(n_samples)
        final_predictions[rows_i, y_pred - 1] += alpha

    # get the final predicted class for each test example
    final_predictions = np.argmax(final_predictions, axis=1) + 1

    # calculate the metrics
    if y_test is not None:
        accuracy = accuracy_score(y_test, final_predictions)
        precision = precision_score(y_test, final_predictions, average="weighted")
        recall = recall_score(y_test, final_predictions, average="weighted")
        f1 = f1_score(y_test, final_predictions, average="weighted")
        return accuracy, precision, recall, f1
    else:
        # not good practice to return different types, but I don't know what the other option is!
        return final_predictions

