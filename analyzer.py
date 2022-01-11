import utils
import data_loader as loader
from pathlib import Path
# from issue_linker import extract_commit_code_terms, extract_commit_text_terms_parts,extract_text
import csv
import matplotlib.pyplot as plt
from collections import Counter
import math
import numpy as np
import data_preprocessor
from feature_options import ExperimentOption
from sklearn.metrics import f1_score, precision_score, recall_score

def get_records_from_ids(ids, records):
    result = []
    id_set = set()

    for id in ids:
        id_set.add(id)

    for record in records:
        if str(record.id) in id_set:
            result.append(record)
    return result


def resolve_path(path):
    base_path = Path(__file__).parent
    file_path = (base_path / path).resolve()

    return file_path


def print_false_case():
    records = loader.load_records("MSR2019/experiment/full_dataset_with_all_features.txt")
    message_false_positives = get_records_from_ids(utils.read_lines(resolve_path("MSR2019/experiment/false_positive/message.txt")), records)
    message_false_negatives = get_records_from_ids(utils.read_lines(resolve_path("MSR2019/experiment/false_negative/message.txt")), records)
    issue_false_positives = get_records_from_ids(utils.read_lines(resolve_path("MSR2019/experiment/false_positive/issue.txt")), records)
    issue_false_negatives = get_records_from_ids(utils.read_lines(resolve_path("MSR2019/experiment/false_negative/issue.txt")), records)
    patch_false_positives = get_records_from_ids(utils.read_lines(resolve_path("MSR2019/experiment/false_positive/patch.txt")), records)
    patch_false_negatives = get_records_from_ids(utils.read_lines(resolve_path("MSR2019/experiment/false_negative/patch.txt")), records)

    for record in message_false_negatives:
        print(record)
        print(record.repo + "/commit/" + record.commit_id)


def write_record_statistics():
    file_name = "record_statistics.csv"
    records = loader.load_records("MSR2019/experiment/full_dataset_with_all_features.txt")
    print(len(records))
    with open(file_name, mode='w') as csv_file:
        fields_names = ['record_id', 'message_length', 'num_code_terms', 'num_text_terms_parts',
                        'min_length_text_terms', 'max_length_text_terms','avg_length_text_terms']

        writer = csv.writer(csv_file)
        writer.writerow(fields_names)

        for record in records:
            record.code_terms = extract_commit_code_terms(record)
            record.text_terms_parts = extract_commit_text_terms_parts(record)
            record.commit_message = extract_text(record.commit_message)

            length_commit_message = len(record.commit_message.split(" "))
            num_code_terms = len(record.code_terms.split(" "))
            num_text_terms_part = len(record.text_terms_parts)
            min_length_text = 999999999
            max_length_text = -1
            avg_length_text = 0
            for part in record.text_terms_parts:
                tokens = part.split(" ")
                min_length_text = min(min_length_text, len(tokens))
                max_length_text = max(max_length_text, len(tokens))
                avg_length_text += len(tokens)

            avg_length_text = int(avg_length_text/len(record.text_terms_parts))

            writer.writerow([record.id, length_commit_message, num_code_terms, num_text_terms_part, min_length_text,
                             max_length_text, avg_length_text])


def plot_most_common_data(items, top_value, title, xlabel, ylabel, step):
    sorted(items)
    c = Counter(items)
    c = c.most_common(top_value)
    c = sorted(c, key=lambda x: x[0])

    x = []
    y = []
    for first, second in c:
        x.append(first)
        y.append(second)

    yint = range(min(y), math.ceil(max(y)) + 1, step)

    plt.yticks(yint)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.plot(x, y)
    plt.scatter(x,y)
    plt.show()


def show_record_statistics():
    file_name = "record_statistics.csv"
    message_length_list = []
    num_code_terms = []
    average_text_terms = []
    min_text_terms = []
    max_text_terms = []
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                continue
            message_length_list.append(row[1])
            num_code_terms.append(row[2])
            average_text_terms.append(row[6])
            min_text_terms.append(row[4])
            max_text_terms.append(row[5])
    plot_most_common_data(message_length_list, 20, "Top message length per record", "", "")
    plot_most_common_data(num_code_terms, 20, "Top number of code terms per record", "", "")
    plot_most_common_data(average_text_terms, 20, "Top average text terms per record", "", "")
    plot_most_common_data(min_text_terms, 20, "Top min text terms per record", "", "")
    plot_most_common_data(max_text_terms, 20, "Top max text terms per record", "", "")

# records = loader.load_records("MSR2019/experiment/full_dataset_with_all_features.txt")
# id_to_rows = {}
# with open('MSR2019/experiment/full_dataset_new.csv') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     count = 0
#
#     for row in csv_reader:
#         count += 1
#         id_to_rows[row[0]] = row
#     print(count)

# new_rows = []
# for record in records:
#     new_rows.append(id_to_rows[record.id])
#
# new_rows.sort(key=lambda x: int(x[0]))
# count = 0
# with open('MSR2019/experiment/full_dataset_new.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file)
#     for row in new_rows:
#         writer.writerow(row)
#         count += 1
# print(count)

# show_record_statistics()



def compare_sim_scores():
    id_to_score_limit_features = {}
    id_to_score_code_terms_only = {}
    id_to_limit_ticket = {}
    id_to_limit_score = {}
    id_to_chunk_ticket = {}
    id_to_chunk_score = {}

    limit_scores = []
    chunk_scores = []
    for lines in utils.read_lines(resolve_path("texts/sim_scores_limit_feature_07042021.txt")):
        parts = lines.split("\t\t")
        record_id = parts[0]
        ticket_id = parts[2]
        score = parts[3]
        id_to_limit_ticket[record_id] = ticket_id
        id_to_limit_score[record_id] = float(score)
        limit_scores.append(round(float(score) * 100))

    plot_most_common_data(limit_scores, 50, "Top 20 similarity scores using linker with nlp terms + code terms", "similarity score (percent)", "Number of records")

    for lines in utils.read_lines(resolve_path("texts/sim_scores_limit_feature_chunk_30.txt")):
        parts = lines.split("\t\t")
        record_id = parts[0]
        ticket_id = parts[2]
        score = parts[3]
        id_to_chunk_ticket[record_id] = ticket_id
        id_to_chunk_score[record_id] = float(score)
        chunk_scores.append(round(float(score) * 100))

    plot_most_common_data(chunk_scores, 50, "Top 20 similarity scores using segmented issues", "similarity score (percent)", "Number of records")
    same_count = 0
    for key, value in id_to_limit_ticket.items():
        if id_to_chunk_ticket[key] == value:
            same_count += 1

    limit_scores = sorted(limit_scores, reverse=True)
    chunk_scores = sorted(chunk_scores, reverse=True)
    print(chunk_scores[1000])
    print(chunk_scores[2000])
    print(chunk_scores[int(0.9 * len(chunk_scores))])
    print(limit_scores[1000])
    print(limit_scores[2000])
    print(limit_scores[int(0.9 * len(limit_scores))])
    print(same_count)
    count_greater = 0
    for key, value in id_to_limit_score.items():
        if id_to_chunk_score[key] >= value:
            count_greater += 1
    print(count_greater)


def analyze_issue_classifier():
    records = loader.load_records("MSR2019/experiment/full_dataset_with_all_features.txt")

    issue_false_positives = get_records_from_ids(utils.read_lines(
            resolve_path("MSR2019/experiment/statistics/false_positive/issue_2021-03-25_19_11_39.547566.txt")), records)
    issue_false_negatives = get_records_from_ids(utils.read_lines(
            resolve_path("MSR2019/experiment/statistics/false_negative/issue_2021-03-25_19_11_39.547566.txt")), records)

    count_jira = 0
    count_github = 0
    # for record in issue_false_negatives:
    #     if len(record.jira_ticket_list) > 0:
    #         count_jira +=1
    #         for ticket in record.jira_ticket_list:
    #             print(record.label)
    #             print(record.repo + '/commit/' + record.commit_id)
    #             print(record.commit_message)
    #             print(ticket.name)
    #             print(ticket.summary)
    #             print(ticket.description)
    #     if len(record.github_issue_list) > 0:
    #         for issue in record.github_issue_list:
    #             print("{} \n {}".format(issue.title, issue.body))
    #         count_github += 1
    #
    #     print("--------------------")
    #
    # print(count_jira)
    # print(count_github)
    # print(len(issue_false_negatives))
    count = 0
    for record in records:
        if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
            count += 1
    print(count)
    # false_positive_scores = []
    # false_negative_scores = []
    #
    # for lines in utils.read_lines(resolve_path("sim_scores_limit_feature_07042021.txt")):
    #     parts = lines.split("\t\t")
    #     record_id = parts[0]
    #     score = round(float(parts[3]) * 100)
    #     if record_id in issue_false_positives:
    #         false_positive_scores.append(score)
    #     if record_id in issue_false_negatives:
    #         false_negative_scores.append(score)
    #
    # plot_data(false_positive_scores, 200, "Issue false positive cases based on similarity scores", "Score in percent", "Number of cases")
    # plot_data(false_negative_scores, 200, "Issue false negative cases based on similarity scores", "Score in percent", "Number of cases")


def analyze_terms():
    records = loader.load_records("MSR2019/experiment/enhanced_dataset_08042021_without_comments_th_80.txt")
    options = ExperimentOption()
    terms_set = {'clamd', 'sb', 'white', 'unmarshal', 'mario', 'fast', 'brkyvz'}
    new_records = []
    count = 0

    for record in records:
        new_records.append(data_preprocessor.preprocess_single_record(record, options))
        count += 1
        if count % 100 == 0:
            print(count)

    records = new_records
    term_to_record_count = {}

    for term in terms_set:
        term_to_record_count[term] = 0
    for record in records:
        for term in terms_set:
            if term in record.issue_info:
                term_to_record_count[term] += 1

    for key, value in term_to_record_count.items():
        print("{}       {}".format(key, value))


def get_jira_repo():
    repo_set = set()
    for lines in utils.read_lines('repo_to_jira.txt'):
        parts = lines.split('\t\t')
        repo = parts[0]
        jira = parts[1]
        if jira == 'https://issues.apache.org/jira':
            repo_set.add(repo)

    for repo in repo_set:
        print(repo)

def get_veracode_score():
    records = loader.load_records("MSR2019/experiment/full_dataset_with_all_features.txt")
    id_to_test = {}
    id_to_pred = {}
    for record in records:
        id_to_test[record.id] = record.label
    miss_data_count = 0
    with open('texts/smu_19_APR.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            record_id = row[1]
            score = float(row[2])
            label = 0
            if score > 0.65:
                print(row)
                label = 1
                # print(row)
            id_to_pred[record_id] = label
    preds = []
    tests = []
    for key, test in id_to_test.items():
        if key not in id_to_pred:
            miss_data_count += 1
            continue
        tests.append(test)
        preds.append(id_to_pred[key])

    precision = precision_score(y_true=tests, y_pred=preds)
    recall = recall_score(y_true=tests, y_pred=preds)
    f1 = f1_score(y_true=tests, y_pred=preds)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("f1: {}".format(f1))
    print("Number of miss data: {}".format(miss_data_count))


def get_veracode_score_2():
    tests = []
    preds = []
    message_to_test_score = {}
    message_to_pred_score = {}
    with open('texts/full_dataset_fixed.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            label = row[4]
            message = row[3]
            if label == 'pos':
                message_to_test_score[message] = 1
            if label == 'neg':
                message_to_test_score[message] = 0


    miss_data_count = 0
    with open('texts/scores.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            score = float(row[3])
            label = 0
            if score > 0.5:
                label = 1
                # print(row)
            message = row[1]
            message_to_pred_score[message] = label

    for key, test_label in message_to_test_score.items():
        if key not in message_to_pred_score:
            miss_data_count += 1
            continue
        tests.append(test_label)
        preds.append(message_to_pred_score[key])

    precision = precision_score(y_true=tests, y_pred=preds)
    recall = recall_score(y_true=tests, y_pred=preds)
    f1 = f1_score(y_true=tests, y_pred=preds)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("f1: {}".format(f1))
    print("Number of miss data: {}".format(miss_data_count))


def analyze_true_link_test():
    records = loader.load_records("MSR2019/experiment/full_dataset_with_all_features.txt")
    count = 0
    pos = 0
    neg = 0
    for record in records:
        if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
            count += 1
            if record.label == 0:
                neg +=1
            if record.label == 1:
                pos +=1
    print("Count: {}".format(count))
    print(pos)
    print(neg)
    id_to_record = {}
    ticket_to_record = {}
    no_link = 0
    for record in records:
        id_to_record[record.id] = record
        if len(record.jira_ticket_list) > 0:
            ticket_id = record.jira_ticket_list[0].name
            ticket_to_record[ticket_id] = record
    correct_link = 0
    total_link = 0
    correct_link_scores = []
    false_link_scores = []
    line_score = []
    correct_label = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_link_positive = 0
    true_link_negative = 0
    for line in utils.read_lines('texts/score_test_new.txt'):
        total_link += 1
        parts = line.split("\t\t")
        record_id = parts[0]
        ticket_id = parts[2]
        score = float(parts[3])
        ticket_key = parts[4]
        record = id_to_record[record_id]
        true_ticket = record.jira_ticket_list[0]
        true_label = record.label
        link_label = None
        if ticket_key != 'None':
            link_label = ticket_to_record[ticket_key].label
        if true_ticket.name == ticket_key:
            correct_link += 1
            correct_link_scores.append(score * 100)
            if true_label == 1:
                true_link_positive += 1
            else:
                true_link_negative += 1
        else:
            false_link_scores.append(score * 100)
            line = line + '\t\t' + true_ticket.name

            if link_label is not None and true_label == link_label:
                correct_label += 1
                if true_label == 1:
                    true_positive += 1
                if true_label == 0:
                    true_negative += 1
            else:
                line_score.append(tuple((line, score)))
                if true_label == 1 and link_label == 0:
                    false_negative += 1
                if true_label == 0 and link_label == 1:
                    false_positive += 1

    line_score.sort(key=lambda x: x[1], reverse=True)
    correct_link_scores = sorted(correct_link_scores)
    false_link_scores = sorted(false_link_scores)

    for line, score in line_score:
        print(line)

    print("Total link: {}".format(total_link))
    print("Correct link: {}".format(correct_link))
    false_link_count = total_link - correct_link
    print("False link count: {}".format(false_link_count))
    print("False link correct label: {}".format(correct_label))
    print("True positive: {}".format(true_positive))
    print("True negative: {}".format(true_negative))
    print("False positive: {}".format(false_positive))
    print("False negative: {}".format(false_negative))
    print("True link positive: {}".format(true_link_positive))
    print("True link negative: {}".format(true_link_negative))
    print("Total correct label: {}".format(correct_link + correct_label))

    # print(correct_link_scores)
    # plot_most_common_data(correct_link_scores, 100,
    #                       "The number of correct linked records correspond to different scores", "Score (percent)", "Number of records")
    # plot_most_common_data(false_link_scores, 100,
    #                       "The number of false linked records correspond to different scores", "Score (percent)", "Number of records")


def do_something():
    records = loader.load_records("MSR")
    print(len(records))
    count = 0
    for record in records:
        if len(record.jira_ticket_list) > 0 or len(record.github_issue_list) > 0:
            count += 1
    print(count)

# do_something()