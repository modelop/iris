from __future__ import print_function

import argparse
import pandas as pd

from sklearn import tree
import pickle


def begin():

    global model
    model = pickle.load(open('model.pkl', 'rb'))



def train(train_data):


    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    max_leaf_nodes = 30

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients

    pickle.dump(clf, open('model.pkl', 'wb'))




def metrics(df):
    y_test = df.iloc[:, 0]
    X_test = df.iloc[:, 1:]
    yield { "ACCURACY": model.score(X_test, y_test)}


def action(X):
    df = pd.DataFrame(X)
    y_pred = model.predict(df)
    for y in y_pred:
        yield y



if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    pred_df =  pd.read_csv('data/test.csv')


    train(train_df)
    begin()


    for m in metrics(train_df):
        print(m)

    for a in action(pred_df):
        print(a)
