import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import click


def fit_probability(x_train, x_test, y_train, y_test):
    ensemble_classifier = LogisticRegression()
    ensemble_classifier.fit(X=x_train, y=y_train)
    y_pred = ensemble_classifier.predict(X=x_test)

    precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
    recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
    f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

    return precision, recall, f1


def fit_simple_voting(x_train, x_test, y_train, y_test):
    y_pred = []
    for prob1,prob2 in x_test:
        # prob1 = probs[0]
        # prob2 = probs[1]
        flag1 = -1
        flag2 = -1
        if prob1 > 0.5:
            flag1 = 0
        if prob1 <= 0.5:
            flag1 = 1

        if prob2 > 0.5:
            flag2 = 0
        if prob2 <= 0.5:
            flag2 = 1
        if flag1 == -1 or flag2 == -1:
            print('error')
        if flag1 == 1 or flag2 == 1:
            y_pred.append(1)
        else:
            y_pred.append(0)

    precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
    recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
    f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

    return precision, recall, f1


@click.command()
@click.option('-type', multiple=True, required=True, type=int)
#type 1: message, 2: issue 3:patch
def calculate_joint_model(type):
    print(type)
    directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = 'classifier_output'
    folder_path = os.path.join(directory, folder_name)

    precisions = []
    recalls = []
    f1s = []

    for file_name in os.listdir(folder_path):
        # print(file_name)
        if file_name == 'dump.txt':
            continue
        with open(folder_path + '/' + file_name) as file:
            content = file.read()
            parts = content.split("@@\n")
            x_train_raw = parts[0]
            x_train_raw = x_train_raw.split("\n")
            x_train_raw = x_train_raw[:(len(x_train_raw) - 1)]
            x_train = []
            for x in x_train_raw:
                items = x.split("\t\t")
                prob1 = float(items[0])
                prob2 = float(items[1])
                prob3 = float(items[2])
                input = []
                if 1 in type:
                    input.append(prob1)
                if 2 in type:
                    input.append(prob2)
                if 3 in type:
                    input.append(prob3)
                x_train.append(input)

            x_test = []
            x_test_raw = parts[1]
            x_test_raw = x_test_raw.split("\n")
            x_test_raw = x_test_raw[:(len(x_test_raw) - 1)]
            for x in x_test_raw:
                items = x.split("\t\t")
                prob1 = float(items[0])
                prob2 = float(items[1])
                prob3 = float(items[2])
                input = []
                if 1 in type:
                    input.append(prob1)
                if 2 in type:
                    input.append(prob2)
                if 3 in type:
                    input.append(prob3)
                x_test.append(input)

            y_train = []
            y_train_raw = parts[2]
            y_train_raw = y_train_raw.split("\n")
            y_train_raw = y_train_raw[:(len(y_train_raw) - 1)]
            for y in y_train_raw:
                y_train.append(int(y))

            y_test = []
            y_test_raw = parts[3]
            y_test_raw = y_test_raw.split("\n")
            y_test_raw = y_test_raw[:(len(y_test_raw) - 1)]
            for y in y_test_raw:
                y_test.append(int(y))

            precision, recall, f1 = fit_probability(x_train, x_test, y_train, y_test)
            # precision, recall, f1 = fit_simple_voting(x_train, x_test, y_train, y_test)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # print("************")

    print("Joint-model mean precision: {}".format(np.mean(precisions)))
    print("Joint-model mean recall: {}".format(np.mean(recalls)))
    print("Joint-model mean f1: {}".format(np.mean(f1s)))


if __name__ == '__main__':
    calculate_joint_model()