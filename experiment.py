from loader import data_loader
from utils import print_line_seperator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
import numpy as np
import random
import data_preprocessor
import feature_options
import click
import utils
import pandas as pd
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import os

np.random.seed(109)

file_path = 'MSR2019/experiment/full_dataset_with_all_features.txt'
options = feature_options.ExperimentOption()


def preprocess_data(records, options):
    print("Start preprocessing commit messages and commit file patches...")
    records = [data_preprocessor.preprocess_single_record(record, options) for record in records]
    print("Finish preprocessing commit messages commit file patches...")
    return records


def filter_using_tf_idf_threshold(records, options):
    print("Filtering using tf-idf threshold...")
    issue_tfidf_vectorizer = TfidfVectorizer(min_df=options.min_document_frequency)
    issue_corpus = []
    record_to_corpus_id = {}
    corpus_count = -1
    for record in records:
        if record.issue_info is not None and record.issue_info != '':
            corpus_count += 1
            issue_corpus.append(record.issue_info)
            record_to_corpus_id[record.id] = corpus_count

    tfidf_matrix = issue_tfidf_vectorizer.fit_transform(issue_corpus)

    feature_names = issue_tfidf_vectorizer.get_feature_names()
    for record in records:
        if record.issue_info is not None and record.issue_info != '':

            # get tf-idf score for every word in document
            doc = record_to_corpus_id[record.id]
            feature_index = tfidf_matrix[doc, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
            token_to_tfidf = {}
            for token, value in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                token_to_tfidf[token] = value

            # generate new issue info contains only valuable terms
            new_issue_info = ''
            for token in record.issue_info.split(' '):
                if token in token_to_tfidf and token_to_tfidf[token] >= options.tf_idf_threshold:
                    new_issue_info = new_issue_info + token + ' '

            record.issue_info = new_issue_info

    print("Finish filtering using tf-idf threshold...")
    return records


def calculate_vocabulary(records, train_data_indices, commit_message_vectorizer, issue_vectorizer,
                         patch_vectorizer, options):

    if options.use_issue_classifier and options.tf_idf_threshold != -1:
        records = filter_using_tf_idf_threshold(records, options)

    # print("Calculating bag of words for log message in train data only")
    commit_message_vectorizer.fit([records[index].commit_message for index in train_data_indices])

    if options.use_issue_classifier:
        issue_vectorizer.fit([records[index].issue_info for index in train_data_indices if records[index].issue_info != ''])

    # print("Calculating bag of words for patch in train data only")
    patch_corpus = []
    for index in train_data_indices:
        record = records[index]
        patch_corpus.append(' '.join([file.patch for file in record.commit.files if file.patch is not None]))
    patch_vectorizer.fit(patch_corpus)


def retrieve_false_positive_negative(y_pred, y_test):
    false_positives = []
    false_negatives = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            false_positives.append(i)
        if y_pred[i] == 0 and y_test[i] == 1:
            false_negatives.append(i)

    return false_positives, false_negatives


def svm_classify(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    train_predict_prob = []
    for predict_prob in classifier.predict_proba(x_train):
        train_predict_prob.append(predict_prob[0])

    test_predict_prob = []
    for predict_prob in classifier.predict_proba(x_test):
        test_predict_prob.append(predict_prob[0])

    false_positives, false_negatives = retrieve_false_positive_negative(y_pred, y_test)

    return metrics.precision_score(y_true=y_test, y_pred=y_pred), \
           metrics.recall_score(y_true=y_test, y_pred=y_pred), \
           metrics.f1_score(y_true=y_test, y_pred=y_pred), y_pred, train_predict_prob, test_predict_prob, false_positives, false_negatives


def retrieve_label(records):
    target = [record.label for record in records]
    target = np.array(target)
    return target


def calculate_log_message_feature_vector(records, commit_message_vectorizer):
    commit_message_features = [commit_message_vectorizer.transform([record.commit_message]).toarray()[0]
                               for record in records]
    commit_message_features = np.array(commit_message_features)

    return commit_message_features, retrieve_label(records)


def calculate_issue_feature_vector(records, issue_vectorizer):
    issue_features = [issue_vectorizer.transform([record.issue_info]).toarray()[0]
                      for record in records if record.issue_info != '']

    issue_features = np.array(issue_features)

    records_with_issue_info = [record for record in records if record.issue_info != '']

    return issue_features, retrieve_label(records_with_issue_info)


def get_join_patch(record):
    result = ' '.join([file.patch for file in record.commit.files if file.patch is not None])
    if result is None:
        return ' '
    return result


def calculate_patch_feature_vector(records, patch_vectorizer):
    patch_features = [patch_vectorizer.transform([get_join_patch(record)]).toarray()[0]
                      for record in records]
    patch_features = np.array(patch_features)

    target = [record.label for record in records]
    target = np.array(target)

    return patch_features, target


def log_message_classify(classifier, x_train, y_train, x_test, y_test):
    # print("Start log message classification...")
    precision, recall, f1, log_message_pred, log_message_train_predict_prob, log_message_test_predict_prob, false_positives, false_negatives \
        = svm_classify(classifier, x_train, x_test, y_train, y_test)

    # print("Precision: {}".format(precision))
    # print("Recall: {}".format(recall))
    return precision, recall, f1, log_message_pred, log_message_train_predict_prob, log_message_test_predict_prob, false_positives, false_negatives


def issue_classify(classifier, x_train, y_train, x_test, y_test, train_data, test_data):
    precision, recall, f1, issue_pred, issue_train_predict_prob, issue_test_predict_prob, false_positives, false_negatives\
        = svm_classify(classifier, x_train, x_test, y_train, y_test)

    id_to_issue_train_predict_prob = {}
    index = 0
    for record in train_data:
        if record.issue_info != '':
            id_to_issue_train_predict_prob[record.id] = issue_train_predict_prob[index]
            index += 1


    id_to_issue_test_predict_prob = {}
    index = 0
    for record in test_data:
        if record.issue_info != '':
            id_to_issue_test_predict_prob[record.id] = issue_test_predict_prob[index]
            index += 1

    return precision, recall, f1, \
           issue_pred, id_to_issue_train_predict_prob, id_to_issue_test_predict_prob, \
           false_positives, false_negatives


def patch_classify(classifier, x_train, y_train, x_test, y_test):
    # print("Start patch classification...")

    precision, recall, f1, patch_pred, patch_train_predict_prob, patch_test_predict_prob, false_positives, false_negatives\
        = svm_classify(classifier, x_train, x_test, y_train, y_test)
    # print("Precision: {}".format(precision))
    # print("Recall: {}".format(recall))
    return precision, recall, f1, patch_pred, patch_train_predict_prob, patch_test_predict_prob, false_positives, false_negatives


def retrieve_data(records, train_data_indices, test_data_indices):
    train_data = [records[index] for index in train_data_indices]
    test_data = [records[index] for index in test_data_indices]

    return train_data, test_data


def measure_joint_model(log_message_prediction, issue_prediction, patch_prediction,
                        log_message_test_predict_prob, patch_test_predict_prob,
                        test_data_labels, options):
    join_prediction = []

    for index in range(len(log_message_prediction)):
        if options.use_issue_classifier:
            join_prediction.append(int(log_message_prediction[index] or patch_prediction[index] or issue_prediction[index]))
        else:
            join_prediction.append(int(log_message_prediction[index] or patch_prediction[index]))

    # precision = metrics.precision_score(y_pred=join_prediction, y_true=test_data_labels)
    # recall = metrics.recall_score(y_pred=join_prediction, y_true=test_data_labels)
    # f1 = metrics.f1_score(y_pred=join_prediction, y_true=test_data_labels)
    log_neg_probs = log_message_test_predict_prob
    log_pos_probs = [1 - prob for prob in log_neg_probs]
    patch_neg_probs = patch_test_predict_prob
    patch_pos_probs = [1 - prob for prob in patch_neg_probs]

    y_pos_probs = []
    for i, log_prob in enumerate(log_pos_probs):
        y_pos_probs.append(max(log_prob, patch_pos_probs[i]))

    y_neg_probs = []
    for i, log_prob in enumerate(log_neg_probs):
        y_neg_probs.append(max(log_prob, patch_neg_probs[i]))


    precision = metrics.precision_score(y_pred=join_prediction, y_true=test_data_labels)
    recall = metrics.recall_score(y_pred=join_prediction, y_true=test_data_labels)
    f1 = metrics.f1_score(y_pred=join_prediction, y_true=test_data_labels)
    auc_roc = metrics.roc_auc_score(y_true=test_data_labels, y_score=y_pos_probs)
    auc_pr = metrics.average_precision_score(y_true=test_data_labels, y_score=y_pos_probs)

    return precision, recall, f1, auc_roc, auc_pr


def measure_joint_model_using_logistic_regression(train_data, test_data, log_message_train_predict_prob, id_to_issue_train_predict_prob,
                                                  patch_train_predict_prob, log_message_test_predict_prob, id_to_issue_test_predict_prob,
                                                  patch_test_predict_prob, options, output_file_name):
    ensemble_classifier = LogisticRegression()

    issue_train_mean_probability = None
    issue_test_mean_probability = None

    if options.use_issue_classifier:
        issue_train_mean_probability = np.mean([prob for id, prob in id_to_issue_train_predict_prob.items()])
        issue_test_mean_probability = np.mean([prob for id, prob in id_to_issue_test_predict_prob.items()])


    y_train = retrieve_label(train_data)
    X_train = []
    lines = ""
    for index in range(len(train_data)):
        if options.use_issue_classifier:
            if train_data[index].id in id_to_issue_train_predict_prob:
                X_train.append([log_message_train_predict_prob[index],
                                id_to_issue_train_predict_prob[train_data[index].id],
                                patch_train_predict_prob[index]])
                lines = lines + str(log_message_train_predict_prob[index]) + '\t\t' \
                        + str(id_to_issue_train_predict_prob[train_data[index].id]) \
                        + '\t\t' + str(patch_train_predict_prob[index]) + '\n'
            else:
                X_train.append(
                    [log_message_train_predict_prob[index],
                     issue_train_mean_probability,
                     patch_train_predict_prob[index]])
        else:
            X_train.append([log_message_train_predict_prob[index], patch_train_predict_prob[index]])

    lines = lines + "@@\n"
    y_test = retrieve_label(test_data)
    X_test = []
    for index in range(len(test_data)):
        if options.use_issue_classifier:
            if test_data[index].id in id_to_issue_test_predict_prob:
                X_test.append(
                    [log_message_test_predict_prob[index],
                     id_to_issue_test_predict_prob[test_data[index].id],
                     patch_test_predict_prob[index]])
                lines = lines + str(log_message_test_predict_prob[index]) \
                        + '\t\t' + str(id_to_issue_test_predict_prob[test_data[index].id]) \
                        + '\t\t' + str(patch_test_predict_prob[index]) + '\n'
            else:
                X_test.append(
                    [log_message_train_predict_prob[index],
                     issue_test_mean_probability,
                     patch_train_predict_prob[index]])
        else:
            X_test.append([log_message_test_predict_prob[index], patch_test_predict_prob[index]])

    lines = lines + "@@\n"

    ensemble_classifier.fit(X=X_train, y=y_train)
    y_pred = ensemble_classifier.predict(X=X_test)
    y_prob = ensemble_classifier.predict_proba(X=X_test)[:, 1]
    joint_precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
    joint_recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
    joint_f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
    joint_auc_roc = metrics.roc_auc_score(y_true=y_test, y_score=y_prob)
    joint_auc_pr = metrics.average_precision_score(y_true=y_test, y_score=y_prob)
    false_positives, false_negatives = retrieve_false_positive_negative(y_pred=y_pred, y_test=y_test)

    for label in y_train:
        lines = lines + str(label) + '\n'
    lines = lines + "@@\n"

    for label in y_test:
        lines = lines + str(label) + '\n'
    return joint_precision, joint_recall, joint_f1, joint_auc_roc, joint_auc_pr, false_positives, false_negatives, lines


def get_list_value_from_string(input):
    return list(map(float, input.strip('[]').split(',')))


def retrieve_word_frequency():
    # do_experiment(get_list_value_from_string(sys.argv[1]))
    records = data_loader.load_records(file_path)
    records = preprocess_data(records, options)

    vectorizer = CountVectorizer()
    transformed_data = vectorizer.fit_transform([record.commit_message for record in records])
    words = vectorizer.get_feature_names()

    frequencies = np.ravel(transformed_data.sum(axis=0)).tolist()
    word_frequency_pair_list = []
    for i in range(len(words)):
        if not words[i].isdigit():
            word_frequency_pair_list.append((words[i], frequencies[i]))

    word_frequency_pair_list.sort(key=lambda x: x[1])

    with open('MSR2019/experiment/statistics/term_frequencies.txt', 'w') as file:
        for word, frequencies in word_frequency_pair_list:
            file.write(str(frequencies) + '\t\t' + word + '\n')


def write_false_index_to_file(false_positive_message_records, false_negative_message_records,
                              false_positive_issue_records, false_negative_issue_records, false_positive_patch_records,
                              false_negative_patch_records, false_positive_joint_records, false_negative_joint_records):
    date = datetime.datetime.now().date()
    time = datetime.datetime.now().time()
    time = str(time).replace(":", "_")
    print("Writing false cases at {}".format(str(date) + "_" + str(time)))
    utils.write_lines(false_positive_message_records, "MSR2019/experiment/statistics/false_positive/message_"
                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_negative_message_records, "MSR2019/experiment/statistics/false_negative/message_"
                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_positive_issue_records, "MSR2019/experiment/statistics/false_positive/issue_"
                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_negative_issue_records, "MSR2019/experiment/statistics/false_negative/issue_"
                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_positive_patch_records, "MSR2019/experiment/statistics/false_positive/patch_"
                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_negative_patch_records, "MSR2019/experiment/statistics/false_negative/patch_"

                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_positive_joint_records, "MSR2019/experiment/statistics/false_positive/joint_"
                      + str(date) + "_" + str(time) + ".txt")
    utils.write_lines(false_negative_joint_records, "MSR2019/experiment/statistics/false_negative/joint_"
                      + str(date) + "_" + str(time) + ".txt")


def to_record_ids(false_positives, test_data_indices):
    record_ids = []
    for index in false_positives:
        record_ids.append(test_data_indices[index].id)

    return record_ids


def retrieve_top_features(classifier, vectorizer):
    print("Feature names with co-efficient scores:")
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
    df = pd.DataFrame(coefs_with_fns)
    df.columns = "coefficient", "word"
    df.sort_values(by="coefficient")

    df_pos = df.tail(30)
    df_pos.style.set_caption("security related words")
    print(df_pos.to_string())

    print_line_seperator()

    df_neg = df.head(30)
    df_neg.style.set_caption("security un-related words")
    print(df_neg.to_string())
    print_line_seperator()

@click.command()
@click.option('-s', '--size', default=-1)
@click.option('--ignore_number', default=True)
@click.option('--github_issue', default=True, type=bool)
@click.option('--jira_ticket', default=True, type=bool)
@click.option('--use_comments', default=True, type=bool)
@click.option('-w', '--positive_weights', multiple=True, default=[0.5], type=float)
@click.option('--n_gram', default=1)
@click.option('--min_df', default=1)
@click.option('--use_linked_commits_only', default=False, type=bool)
@click.option('--use_issue_classifier', default=True, type=bool)
@click.option('--fold_to_run', default=10, type=int)
@click.option('--use_stacking_ensemble', default=True, type=bool)
@click.option('--dataset', default='', type=str)
@click.option('--tf-idf-threshold', default=-1, type=float)
@click.option('--use-patch-context-lines', default=False, type=bool)
@click.option('--run-fold', default=-1, type=int)
def do_experiment(size, ignore_number, github_issue, jira_ticket, use_comments, positive_weights, n_gram, min_df,
                  use_linked_commits_only, use_issue_classifier, fold_to_run, use_stacking_ensemble, dataset,
                  tf_idf_threshold, use_patch_context_lines, run_fold):

    global file_path
    if dataset != '':
        file_path = 'MSR2019/experiment/' + dataset

    print("Dataset: {}".format(file_path))

    options = feature_options.read_option_from_command_line(size, 0, ignore_number,
                                                            github_issue, jira_ticket, use_comments,
                                                            positive_weights,
                                                            n_gram, min_df, use_linked_commits_only,
                                                            use_issue_classifier,
                                                            fold_to_run,
                                                            use_stacking_ensemble,
                                                            tf_idf_threshold,
                                                            use_patch_context_lines)

    commit_message_vectorizer = CountVectorizer(ngram_range=(1, options.max_n_gram))

    issue_vectorizer = CountVectorizer(ngram_range=(1, options.max_n_gram),
                                       min_df=options.min_document_frequency)

    patch_vectorizer = CountVectorizer()

    positive_weights = options.positive_weights

    records = data_loader.load_records(file_path)

    random.shuffle(records)

    if options.use_linked_commits_only:
        new_records = []
        for record in records:
            if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
                new_records.append(record)
        records = new_records

    if options.data_set_size != -1:
        records = records[:options.data_set_size]

    records = preprocess_data(records, options)

    k_fold = KFold(n_splits=10, shuffle=True, random_state=109)

    weight_to_log_classifier = {}
    weight_to_patch_classifier = {}
    weight_to_issue_classifier = {}

    weight_to_log_precisions = {}
    weight_to_log_recalls = {}
    weight_to_log_f1s = {}

    weight_to_patch_precisions = {}
    weight_to_patch_recalls = {}
    weight_to_patch_f1s = {}

    weight_to_issue_precisions = {}
    weight_to_issue_recalls = {}
    weight_to_issue_f1s = {}

    weight_to_joint_precisions = {}
    weight_to_joint_recalls = {}
    weight_to_joint_f1s = {}
    weight_to_joint_auc_roc = {}
    weight_to_joint_auc_pr = {}

    for positive_weight in positive_weights:
        negative_weight = 1 - positive_weight
        weights = {1: positive_weight, 0: negative_weight}
        log_classifier = svm.SVC(kernel='linear', class_weight=weights, probability=True)
        patch_classifier = svm.SVC(kernel='linear', class_weight=weights, probability=True)
        issue_classifier = svm.SVC(kernel='linear', class_weight=weights, probability=True)

        weight_to_log_classifier[positive_weight] = log_classifier
        weight_to_patch_classifier[positive_weight] = patch_classifier
        weight_to_issue_classifier[positive_weight] = issue_classifier

        weight_to_log_precisions[positive_weight] = []
        weight_to_log_recalls[positive_weight] = []
        weight_to_log_f1s[positive_weight] = []

        weight_to_patch_precisions[positive_weight] = []
        weight_to_patch_recalls[positive_weight] = []
        weight_to_patch_f1s[positive_weight] = []

        weight_to_issue_precisions[positive_weight] = []
        weight_to_issue_recalls[positive_weight] = []
        weight_to_issue_f1s[positive_weight] = []

        weight_to_joint_precisions[positive_weight] = []
        weight_to_joint_recalls[positive_weight] = []
        weight_to_joint_f1s[positive_weight] = []
        weight_to_joint_auc_roc[positive_weight] = []
        weight_to_joint_auc_pr[positive_weight] = []

    false_positive_message_records = []
    false_negative_message_records = []
    false_positive_issue_records = []
    false_negative_issue_records = []
    false_positive_patch_records = []
    false_negative_patch_records = []
    false_positive_joint_records = []
    false_negative_joint_records = []

    fold_count = 0

    date = datetime.datetime.now().date()
    time = datetime.datetime.now().time()
    time = str(time).replace(":", "_")

    directory = os.path.dirname(os.path.abspath(__file__))

    for train_data_indices, test_data_indices in k_fold.split(records):
        fold_count += 1
        if run_fold != -1 and fold_count != run_fold:
            continue
        output_file_name = "fold_" + str(fold_count) + "_" + str(date) + "_" + str(time) + ".txt"
        output_file_path = os.path.join(directory, "classifier_output/" + output_file_name)
        if fold_count > options.fold_to_run:
            break
        print("Processing fold number: {}".format(fold_count))
        calculate_vocabulary(records, train_data_indices, commit_message_vectorizer,
                             issue_vectorizer, patch_vectorizer, options)

        train_data, test_data = retrieve_data(records, train_data_indices, test_data_indices)

        log_x_train, log_y_train = calculate_log_message_feature_vector(train_data, commit_message_vectorizer)
        log_x_test, log_y_test = calculate_log_message_feature_vector(test_data, commit_message_vectorizer)

        issue_x_train, issue_y_train, issue_x_test, issue_y_test = None, None, None, None

        if options.use_issue_classifier:
            issue_x_train, issue_y_train = calculate_issue_feature_vector(train_data, issue_vectorizer)
            issue_x_test, issue_y_test = calculate_issue_feature_vector(test_data, issue_vectorizer)

        patch_x_train, patch_y_train = calculate_patch_feature_vector(train_data, patch_vectorizer)
        patch_x_test, patch_y_test = calculate_patch_feature_vector(test_data, patch_vectorizer)

        for positive_weight in positive_weights:
            print("Current processing weight set ({},{})".format(positive_weight, 1 - positive_weight))
            log_classifier = weight_to_log_classifier[positive_weight]

            issue_classifier = None
            id_to_issue_train_predict_prob = None
            id_to_issue_test_predict_prob = None
            if options.use_issue_classifier:
                issue_classifier = weight_to_issue_classifier[positive_weight]

            patch_classifier = weight_to_patch_classifier[positive_weight]

            # calculate precision, recall for log message classification
            precision, recall, f1, log_message_prediction, log_message_train_predict_prob, log_message_test_predict_prob, false_positives, false_negatives\
                = log_message_classify(log_classifier, log_x_train, log_y_train, log_x_test, log_y_test)

            print("Message F1: {}".format(f1))
            # print("Top features for log message classifier:")
            # retrieve_top_features(log_classifier, commit_message_vectorizer)
            # print_line_seperator()

            weight_to_log_precisions[positive_weight].append(precision)
            weight_to_log_recalls[positive_weight].append(recall)
            weight_to_log_f1s[positive_weight].append(f1)
            false_positive_message_records.extend(to_record_ids(false_positives, test_data))
            false_negative_message_records.extend(to_record_ids(false_negatives, test_data))

            # calculate precision, recall for issue classification
            precision, recall, f1, issue_prediction = None, None, None, None
            if options.use_issue_classifier:
                precision, recall, f1, issue_prediction, id_to_issue_train_predict_prob, id_to_issue_test_predict_prob, false_positives, false_negatives\
                    = issue_classify(issue_classifier, issue_x_train, issue_y_train, issue_x_test, issue_y_test, train_data, test_data)

                print("Issue F1: {}".format(f1))
                # print("Top features for issue classifier:")
                # retrieve_top_features(issue_classifier, issue_vectorizer)
                # print_line_seperator()

                weight_to_issue_precisions[positive_weight].append(precision)
                weight_to_issue_recalls[positive_weight].append(recall)
                weight_to_issue_f1s[positive_weight].append(f1)
                false_positive_issue_records.extend(to_record_ids(false_positives, test_data))
                false_negative_issue_records.extend(to_record_ids(false_negatives, test_data))


            # calculate precision, recall for patch

            precision, recall, f1, patch_prediction, patch_train_predict_prob, patch_test_predict_prob, false_positives, false_negatives\
                = patch_classify(patch_classifier, patch_x_train, patch_y_train, patch_x_test, patch_y_test)

            print("Patch F1: {}".format(f1))
            # print("Top features for patch classifier:")
            # retrieve_top_features(patch_classifier, patch_vectorizer)
            # print_line_seperator()

            weight_to_patch_precisions[positive_weight].append(precision)
            weight_to_patch_recalls[positive_weight].append(recall)
            weight_to_patch_f1s[positive_weight].append(f1)
            false_positive_patch_records.extend(to_record_ids(false_positives, test_data))
            false_negative_patch_records.extend(to_record_ids(false_negatives, test_data))

            # calculate precision, recall for joint-model
            joint_precision, joint_recall, joint_f1 = None, None, None

            if options.use_stacking_ensemble:
                joint_precision, joint_recall, joint_f1, joint_auc_roc, joint_auc_pr, false_positive_joint_records, false_negative_joint_records, output_lines \
                    = measure_joint_model_using_logistic_regression(train_data=train_data,
                                                                    test_data=test_data,
                                                                    log_message_train_predict_prob=log_message_train_predict_prob,
                                                                    id_to_issue_train_predict_prob=id_to_issue_train_predict_prob,
                                                                    patch_train_predict_prob=patch_train_predict_prob,
                                                                    log_message_test_predict_prob=log_message_test_predict_prob,
                                                                    id_to_issue_test_predict_prob=id_to_issue_test_predict_prob,
                                                                    patch_test_predict_prob=patch_test_predict_prob, options=options,
                                                                    output_file_name = output_file_name)
                false_positive_joint_records.extend(to_record_ids(false_positives, test_data))
                false_negative_joint_records.extend(to_record_ids(false_negatives, test_data))
                with open(output_file_path, 'w') as f:
                    f.write(output_lines)
                f.close()
            else:
                joint_precision, joint_recall, joint_f1, joint_auc_roc, joint_auc_pr \
                    = measure_joint_model(log_message_prediction, issue_prediction,
                                          patch_prediction, log_message_test_predict_prob, patch_test_predict_prob, retrieve_label(test_data), options)

            weight_to_joint_precisions[positive_weight].append(joint_precision)
            weight_to_joint_recalls[positive_weight].append(joint_recall)
            weight_to_joint_f1s[positive_weight].append(joint_f1)
            weight_to_joint_auc_roc[positive_weight].append(joint_auc_roc)
            weight_to_joint_auc_pr[positive_weight].append(joint_auc_pr)

        break
    print_line_seperator()

    for positive_weight in positive_weights:
        print("Training result for positive weight: {}, negative weight: {}".format(positive_weight, 1 - positive_weight))
        print("Log message mean precision: {}".format(np.mean(weight_to_log_precisions[positive_weight])))
        print("Log message mean recall: {}".format(np.mean(weight_to_log_recalls[positive_weight])))
        print("Log message mean f1: {}".format(np.mean(weight_to_log_f1s[positive_weight])))

        if options.use_issue_classifier:
            print("Issue mean precision: {}".format(np.mean(weight_to_issue_precisions[positive_weight])))
            print("Issue mean recall: {}".format(np.mean(weight_to_issue_recalls[positive_weight])))
            print("Issue mean f1: {}".format(np.mean(weight_to_issue_f1s[positive_weight])))

        print("Patch mean precision: {}".format(np.mean(weight_to_patch_precisions[positive_weight])))
        print("Patch mean recall: {}".format(np.mean(weight_to_patch_recalls[positive_weight])))
        print("Patch mean f1: {}".format(np.mean(weight_to_patch_f1s[positive_weight])))

        print("Joint-model mean precision: {}".format(np.mean(weight_to_joint_precisions[positive_weight])))
        print("Joint-model mean recall: {}".format(np.mean(weight_to_joint_recalls[positive_weight])))
        print("Joint-model mean f1: {}".format(np.mean(weight_to_joint_f1s[positive_weight])))
        print("Joint-model mean AUC-ROC: {}".format(np.mean(weight_to_joint_auc_roc[positive_weight])))
        print("Joint-model mean AUC-PR: {}".format(np.mean(weight_to_joint_auc_pr[positive_weight])))
        print_line_seperator()

    write_false_index_to_file(false_positive_message_records, false_negative_message_records,
                              false_positive_issue_records, false_negative_issue_records,
                              false_positive_patch_records, false_negative_patch_records,
                              false_positive_joint_records, false_negative_joint_records)


if __name__ == '__main__':
    do_experiment()

# records = loader.load_records(file_path)
# count_issue = 0
# count_ticket = 0
# count_both = 0
# for record in records:
#     if len(record.github_issue_list) > 0:
#         count_issue += 1
#     if len(record.jira_ticket_list) > 0:
#         count_ticket += 1
#     if len(record.github_issue_list) > 0 and len(record.jira_ticket_list) > 0:
#         count_both +=1
#
# print(count_issue)
# print(count_ticket)
# print(count_both)

# count_pos = 0
# count_neg = 0
# count_pos_all = 0
# count_neg_all = 0
# count_other = 0
# records = loader.load_records(file_path)
# for record in records:
#     if record.label == 1:
#         count_pos_all += 1
#     if record.label == 0:
#         count_neg_all += 1
#     if record.label != 0 and record.label != 1:
#         print(record)
#     if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
#         if record.label == 1:
#             count_pos += 1
#         if record.label == 0:
#             count_neg += 1
#
#     if len(record.github_issue_list) == 0 and len(record.jira_ticket_list) == 0:
#         count_other += 1
#
#
# print(count_pos)
# print(count_neg)
# print(count_pos_all)
# print(count_neg_all)
# print(count_other)