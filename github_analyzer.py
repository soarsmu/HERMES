import deep_experiment
import os
from data_loader import data_loader
import analyzer
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import utils
import csv
import pandas as pd


directory = os.path.dirname(os.path.abspath(__file__))

record_file_path = os.path.join(directory, 'MSR2019/experiment/full_dataset_with_all_features.txt')

records = data_loader.load_records(record_file_path)
id_to_record = {}

for record in records:
    id_to_record[int(record.id)] = record

commit_to_username, repo_to_username, repo_to_contributor_data, repo_to_total_count = {}, {}, {}, {}
repo_to_contributor_total_count, repo_to_first_commit_time, repo_to_repo_first_commit_time = {}, {}, {}
repo_to_contributor_rank, record_to_contributor_commit_count, record_to_contributor_latest_commit = {}, {}, {}
commit_to_username = deep_experiment.get_commit_to_username()


def init_data():
    repo_to_username = deep_experiment.get_repo_to_username()
    repo_to_contributor_data, repo_to_total_count, repo_to_contributor_total_count \
        = deep_experiment.get_repo_to_contributor_data(id_to_record)
    repo_to_first_commit_time, repo_to_repo_first_commit_time\
        = deep_experiment.get_repo_to_contributor_first_commit_time(repo_to_contributor_data)
    repo_to_contributor_rank = deep_experiment.get_repo_to_contributor_rank(repo_to_contributor_total_count)
    record_to_contributor_commit_count, record_to_contributor_latest_commit \
        = deep_experiment.get_record_to_contributor_commit_info(records, commit_to_username)


def analyze_q1():
    # His position on top 100 contributors of the project
    pos = []
    neg = []

    for record in records:
        repo = record.repo
        if int(record.id) not in commit_to_username:
            continue
        username = commit_to_username[int(record.id)]
        if username not in repo_to_contributor_rank[repo]:
            continue
        rank = repo_to_contributor_rank[repo][username]
        if rank > 50:
            continue
        if record.label == 1:
            pos.append(repo_to_contributor_rank[repo][username])
        if record.label == 0:
            neg.append(repo_to_contributor_rank[repo][username])

    pos.sort()
    neg.sort()
    plt.hist(x=pos, bins=50)
    plt.title("Committer's rank in repository for positive commits")
    plt.xlabel("Rank")
    plt.ylabel("Number of committers")
    plt.show()
    plt.hist(x=neg, bins=50)
    plt.title("Committer's rank in repository for negative commits")
    plt.ylabel("Number of committers")
    plt.show()
    # analyzer.plot_most_common_data(pos, len(pos), "contributor rank for pos", "rank", "number of commiter", 500)
    # analyzer.plot_most_common_data(neg, len(neg), "contributor rank for neg", "rank", "number of commiter", 500)


def analyze_q2():
    # How many commits he has made on these file in patch? (the file that he has most commit)
    pos = []
    neg = []
    for record in records:
        record_id = int(record.id)
        if record_id not in record_to_contributor_commit_count:
            continue
        commit_count = record_to_contributor_commit_count[record_id]
        if commit_count > 50:
            continue
        if record.label == 1:
            pos.append(commit_count)
        else:
            neg.append(commit_count)

    # analyzer.plot_most_common_data(pos, len(pos), "contributor commit count on relevant files for pos", "number of commit", "counter", 100)
    # analyzer.plot_most_common_data(neg, len(neg), "contributor commit count on relevant files for neg", "number of commit", "counter", 500)
    pos.sort()
    neg.sort()
    plt.hist(x=pos, bins=100)
    plt.title("Commit count relevant files for positive commits")
    plt.xlabel("Number of commits")
    plt.ylabel("Counter")
    plt.show()
    plt.hist(x=neg, bins=100)
    plt.title("Commit count on relevant files for negative commits")
    plt.xlabel("Number of commits")
    plt.ylabel("Counter")
    plt.show()


def analyze_q3():
    # When was the most recent commit that he made on file in patch?
    pos = []
    neg = []
    for record in records:
        record_id = int(record.id)
        commit_datetime = datetime.strptime(record.commit.created_date, "%a, %d %b %Y %H:%M:%S GMT")
        commit_timestamp = time.mktime(commit_datetime.timetuple())
        if record_id not in record_to_contributor_latest_commit:
            continue

        latest_timestamp = record_to_contributor_latest_commit[record_id]
        latest_datetime = datetime.utcfromtimestamp(latest_timestamp)

        day = (commit_timestamp - latest_timestamp) / 3600 / 24
        if day > 100:
            continue
        # if day / 365 > 20:
        #     continue

        if record.label == 1:
            pos.append(day)
        else:
            neg.append(day)
    pos.sort()
    neg.sort()
    plt.hist(x=pos, bins=100)
    plt.title("Recent commit time on relevant files for positive commits")
    plt.xlabel("Number of days")
    plt.ylabel("Counter")
    plt.show()
    plt.hist(x=neg, bins=100)
    plt.title("Recent commit time on relevant files for negative commits")
    plt.xlabel("Number of days")
    plt.ylabel("Counter")
    plt.show()
    # analyzer.plot_most_common_data(pos, len(pos), "Distance to latest commit - pos", "Day", "Counter", 1)
    # analyzer.plot_most_common_data(neg, len(neg), "Distance to latest commit - neg", "Day", "Counter", 1)


# analyze_q3()

def analyze_q4():
    # Number of days from first commit on the project to vulnerability fix commit date
    pos = []
    neg = []

    for record in records:
        repo = record.repo
        record_id = int(record.id)
        commit_datetime = datetime.strptime(record.commit.created_date, "%a, %d %b %Y %H:%M:%S GMT")
        commit_timestamp = time.mktime(commit_datetime.timetuple())
        if record_id not in commit_to_username:
            continue
        username = commit_to_username[record_id]
        if username not in repo_to_first_commit_time[repo]:
            continue

        # if record_id not in record_to_contributor_latest_commit:
        #     continue
        #
        first_commit_timestamp = repo_to_first_commit_time[repo][username]
        if commit_timestamp < first_commit_timestamp:
            continue
        day = (commit_timestamp - first_commit_timestamp) / 3600 / 24
        if day < 0:
            print(first_commit_timestamp)
            print(commit_timestamp)
            print(record.id)
        if day > 50:
            continue
        if record.label == 1:
            pos.append(day)
        else:
            neg.append(day)

    pos.sort()
    neg.sort()
    plt.hist(x=pos, bins=100)
    plt.title("Days from first commit on a repository to a vulnerability fix commit - for positive commits")
    plt.xlabel("Number of days")
    plt.ylabel("Counter")
    plt.show()
    plt.hist(x=neg, bins=100)
    plt.title("Days from first commit on a repository to a vulnerability fix commit - for negative commits")
    plt.xlabel("Number of days")
    plt.ylabel("Counter")
    plt.show()


def do_something():
    repo_to_commit_count = {}
    for record in records:
        repo = record.repo
        if repo not in repo_to_commit_count:
            repo_to_commit_count[repo] = 1
        else:
            repo_to_commit_count[repo] += 1

    repo = None
    count = 0
    for x, y in repo_to_commit_count.items():
        # if x == "https://github.com/apache/tomcat":
        #     continue
        if y > count:
            count = y
            repo = x

    commiter_pos = {}
    for record in records:
        record_id = int(record.id)
        if record.repo == repo:
            if record.label == 1:
                commiter = commit_to_username[record_id]
                print("{}   =>    {}".format(record.label, commiter))
                print("{}".format(record.repo + "/commit/" + record.commit_id))
                if commiter not in commiter_pos:
                    commiter_pos[commiter] = 0
                commiter_pos[commiter] += 1


    commiter_neg = {}
    for record in records:
        record_id = int(record.id)
        if record.repo == repo:
            if record.label == 0:
                commiter = commit_to_username[record_id]
                print("{}   =>    {}".format(record.label, commiter))
                print("{}".format(record.repo + "/commit/" + record.commit_id))
                if commiter not in commiter_neg:
                    commiter_neg[commiter] = 0
                commiter_neg[commiter] += 1

    print("Positive committer...")
    for x, y in commiter_pos.items():
        print(x, y)
    print("------------------------------")
    print("Negative committer...")
    for x, y in commiter_neg.items():
        print(x, y)

    for record in records:
        print(record.repo + '/commit/' + record.commit_id)
        print(record.commit_message)
        print("----------")


def analyze_commit_files():
    repo_to_commit_files = {}
    pos_repo_to_commit_files = {}
    neg_repo_to_commit_files = {}

    for record in records:
        repo = record.repo
        if repo not in repo_to_commit_files:
            repo_to_commit_files[repo] = {}
        if repo not in pos_repo_to_commit_files:
            pos_repo_to_commit_files[repo] = {}
        if repo not in neg_repo_to_commit_files:
            neg_repo_to_commit_files[repo] = {}

        commit = record.commit
        for file in commit.files:
            if file.file_name not in repo_to_commit_files[repo]:
                repo_to_commit_files[repo][file.file_name] = 0
            repo_to_commit_files[repo][file.file_name] += 1

        if record.label == 1:
            for file in commit.files:
                if file.file_name not in pos_repo_to_commit_files[repo]:
                    pos_repo_to_commit_files[repo][file.file_name] = 0
                pos_repo_to_commit_files[repo][file.file_name] += 1

        if record.label == 0:
            for file in commit.files:
                if file.file_name not in neg_repo_to_commit_files[repo]:
                    neg_repo_to_commit_files[repo][file.file_name] = 0
                neg_repo_to_commit_files[repo][file.file_name] += 1

    pos_file_name_set = set()
    for repo in pos_repo_to_commit_files:
        # print(repo)
        for file_name, count in pos_repo_to_commit_files[repo].items():
            if count >= 3:
                parts = file_name.split("/")
                file_name = parts[len(parts) - 1].split('.')[0]
                if "test" in file_name.lower():
                    continue
                pos_file_name_set.add(file_name)

    neg_file_name_set = set()
    for repo in neg_repo_to_commit_files:
        # print(repo)
        for file_name, count in neg_repo_to_commit_files[repo].items():
            # if count >= 3:
            parts = file_name.split("/")
            file_name = parts[len(parts) - 1].split('.')[0]
            if "test" in file_name.lower():
                continue
            neg_file_name_set.add(file_name)

    inter_set = pos_file_name_set.intersection(neg_file_name_set)
    print(len(pos_file_name_set))
    print(len(neg_file_name_set))
    print(len(inter_set))
    for file_name in pos_file_name_set:
        if file_name not in inter_set:
            print(file_name)

# do_something()

def analyze_pull_request():
    pos_commiter = set()
    neg_commiter = set()
    data = []

    username_to_pr_count = {}
    file_path = os.path.join(directory, "data/github_statistics/pull_request/509.csv")
    pr_time = []
    pos_time = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            count += 1
            if count == 1:
                continue
            username = row[5]
            if username == "None-":
                continue
            created_time = int(float(row[3]))
            if username == "oscerd":
                pr_time.append(created_time)
            if username not in username_to_pr_count:
                username_to_pr_count[username] = 0
            username_to_pr_count[username] += 1


    for record in records:
        repo = record.repo
        commit_id = int(record.id)
        if utils.clear_github_prefix(repo) != 'apache/camel':
            continue
        if commit_id not in commit_to_username:
            continue
        username = commit_to_username[commit_id]
        if username not in username_to_pr_count:
            continue
        if username == "oscerd" and record.label == 1:
            print(repo + '/commit/' + record.commit_id)
            print(record.commit.created_date)
            commit_time = datetime.strptime(record.commit.created_date, "%a, %d %b %Y %H:%M:%S GMT")
            timestamp = time.mktime(commit_time.timetuple())
            pos_time.append(timestamp)
        data.append((record.id, username, username_to_pr_count[username], record.label))

    df = pd.DataFrame(data=data, columns=['record_id', 'username', 'pr_count', 'label'])

    pd.set_option('display.expand_frame_repr', False)

    # print(df)
    plt.hist(pr_time, bins=100, label="pull request")
    plt.hist(pos_time, bins=100, label="vulnerability fix commit")
    plt.legend(loc='upper right')
    plt.show()

    # pr_time.sort()
    # pos_time.sort()
    # plt.boxplot(pr_time)
    # plt.boxplot(pos_time)
    # plt.show()

    plt.boxplot([pr_time, pos_time])
    # plt.title("Distribution in the number of joined repositories of committers")
    # plt.ylabel("Number of joined repositories")
    plt.xticks([1, 2], ['pr time', 'pos time'])
    plt.show()
    print()


analyze_pull_request()