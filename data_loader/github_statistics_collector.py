import data_loader
import os
import github
import utils
import csv
import requests
import json
import math
import matplotlib.pyplot as plt
from collections import Counter
import analyzer
from datetime import datetime
from time import mktime
import time
import statistics
# import deep_experiment
import csv
import deep_experiment
from entities import GithubCommit, GithubCommitFile, EntityEncoder

directory = os.path.dirname(os.path.abspath(__file__))

labeled_record_file_path = os.path.join(directory, '../MSR2019/experiment/full_dataset_with_all_features.txt')
access_tokens = ['2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24',
                 'dd28467db1a1c81d7dcef7c17ae8bc0d850ef9cf',
                 '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',
                 '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']

max_date = 0
min_date = 999999999999
def get_author_username():
    records = data_loader.load_records(labeled_record_file_path)
    token_index = 0
    repo_to_commits = {}
    commit_to_record_id = {}

    for record in records:
        repo = record.repo
        commit_id = record.commit_id
        if repo not in repo_to_commits:
            repo_to_commits[repo] = []
        repo_to_commits[repo].append(commit_id)
        commit_to_record_id[repo + commit_id] = record.id

    gh = github.Github(access_tokens[token_index])
    processed_repo = set()
    processed_id = set()
    stop = False
    lines = []
    while not stop:
        try:
            for repo_name, commit_ids in repo_to_commits.items():
                if repo_name in processed_repo:
                    continue
                repo = gh.get_repo(utils.clear_github_prefix(repo_name))
                for commit_id in commit_ids:
                    record_id = commit_to_record_id[repo_name + commit_id]
                    if record_id in processed_id:
                        continue
                    commit = repo.get_commit(commit_id)
                    author_login = 'None'
                    if hasattr(commit.author, "login"):
                        author_login = commit.author.login
                    lines.append((record_id, repo_name, commit_id, author_login))
                    processed_id.add(record_id)
                    print("{}   {}    {}".format(record_id, commit_id, author_login))
                print("Finish repo: {}".format(repo_name))
                processed_repo.add(repo_name)
            stop = True
        except github.RateLimitExceededException:
            token_index += 1
            if token_index == 4:
                token_index = 0
            gh = github.Github(access_tokens[token_index])

    with open('commit_username.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for record_id, repo_name, commit_id, author_login in lines:
            csv_writer.writerow([record_id, repo_name, commit_id, author_login])


def get_contributor_activity():
    records = data_loader.load_records(labeled_record_file_path)
    repo_to_id = {}
    for record in records:
        repo = record.repo
        repo_to_id[repo] = record.id

    for repo_name, record_id in repo_to_id.items():
        print(repo_name)
        short_repo = utils.clear_github_prefix(repo_name)
        url = 'https://api.github.com/repos/' + short_repo + '/stats/contributors'
        headers = {'Authorization': 'bearer 2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24'}
        r = requests.get(url, headers=headers)
        absolute_path = '/Users/nguyentruonggiang/Desktop/smu/project/AutoVulCuration'
        file_path = absolute_path + '/data/github_statistics/contributor_activity/' + str(repo_to_id[repo_name]) + '.json'
        with open(file_path, 'w') as file:
            file.write(r.text)


def get_contributor_activity_again():
    folder_path = '/Users/nguyentruonggiang/Desktop/smu/project/AutoVulCuration//data/github_statistics/contributor_activity'

    id_to_repo = {}
    records = data_loader.load_records(labeled_record_file_path)
    for record in records:
        id_to_repo[int(record.id)] = record.repo

    has_empty = False
    for file_name in os.listdir(folder_path):
        with open(folder_path + '/' + file_name, 'r') as reader:
            content = reader.read()
            is_empty = False
            if content == '{}' or len(content) < 5:
                print(file_name)
                has_empty = True
                is_empty = True
            reader.close()
            if is_empty:
                record_id = int(file_name.split('.')[0])
                repo = id_to_repo[record_id]
                short_repo = utils.clear_github_prefix(repo)
                url = 'https://api.github.com/repos/' + short_repo + '/stats/contributors'
                headers = {'Authorization': 'bearer 2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24'}
                r = requests.get(url, headers=headers)
                absolute_path = '/Users/nguyentruonggiang/Desktop/smu/project/AutoVulCuration'
                file_path = absolute_path + '/data/github_statistics/contributor_activity/' + file_name
                with open(file_path, 'w') as file:
                    file.write(r.text)

    print(has_empty)


def map_github_username_to_name():
    contributor_activity_file_path = os.path.join(directory,
                                                  '../data/github_statistics/contributor_activity')
    username_set = set()
    for file_name in os.listdir(contributor_activity_file_path):
        if not file_name.endswith('.json'):
            continue
        with open(contributor_activity_file_path + '/' + file_name, 'r') as reader:
            item_list = json.loads(reader.read())
            for item in item_list:
                username_set.add(item['author']['login'])

    username_to_name = {}
    token_index = 0
    gh = github.Github(access_tokens[token_index])
    stop = False
    while not stop:
        try:
            for username in username_set:
                if username in username_to_name:
                    continue
                try:
                    info = gh.get_user(username)
                    name = 'None'
                    if info.name is not None:
                        name = info.name
                    username_to_name[username] = name
                    print("{}       =>      {}".format(username, name))
                except github.UnknownObjectException:
                    continue
            stop = True
        except github.RateLimitExceededException:
            token_index += 1
            if token_index == 4:
                token_index = 0
            gh = github.Github(access_tokens[token_index])

    with open('../data/github_statistics/username_name.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for username, name in username_to_name.items():
            csv_writer.writerow([username, name])


def clean_commit_username_data():
    commit_username_clean = []
    username_to_name = {}
    name_to_username = {}
    records = data_loader.load_records(labeled_record_file_path)
    id_to_record = {}
    for record in records:
        id_to_record[int(record.id)] = record

    with open('../data/github_statistics/username_name.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        count_duplicate = 0
        for row in csv_reader:
            username_to_name[row[0]] = row[1]
            if row[1] != 'None':
                if row[1] in name_to_username:
                    count_duplicate += 1
                    # print("duplicate name: {}".format(row[1]))
                name_to_username[row[1]] = row[0]
        print('duplicate: {}'.format(count_duplicate))

    username_found = 0
    none_count = 0
    with open('../data/github_statistics/commit_username.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            alias = row[3]
            if not alias.startswith("None-"):
                commit_username_clean.append([row[0], row[1], row[2], row[3]])
            else:
                none_count += 1
                # print(alias)
                alias = alias[len('None-'):]
                if alias in username_to_name:
                    commit_username_clean.append([row[0], row[1], row[2], alias])
                elif alias in name_to_username:
                    commit_username_clean.append([row[0], row[1], row[2], name_to_username[alias]])
                else:
                    commit_username_clean.append([row[0], row[1], row[2], 'None-'])

    with open('commit_username_clean.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in commit_username_clean:
            csv_writer.writerow(row)


def get_commit_username():
    commit_to_username = {}
    with open('../data/github_statistics/commit_username_clean.csv', mode = 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            commit_id = int(row[0])
            username = row[3]
            commit_to_username[commit_id] = username
    return commit_to_username


def analyze_q1():
    records = data_loader.load_records(labeled_record_file_path)
    id_to_record = {}

    commit_to_username = get_commit_username()

    for record in records:
        id_to_record[int(record.id)] = record

    repo_to_username = {}
    with open('../data/github_statistics/commit_username_clean.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            repo = row[1]
            if repo not in repo_to_username:
                repo_to_username[repo] = set()
            username = row[3]
            if username != 'None-':
                repo_to_username[repo].add(username)

    contributor_activity_file_path = os.path.join(directory,
                                                  '../data/github_statistics/contributor_activity')

    repo_to_total_count = {}
    repo_to_contributor_data = {}
    for file_name in os.listdir(contributor_activity_file_path):
        if not file_name.endswith('.json'):
            continue
        record_id = int(file_name.split('.')[0])
        repo = id_to_record[record_id].repo
        repo_to_total_count[repo] = 0
        repo_to_contributor_data[repo] = {}
        with open(contributor_activity_file_path + '/' + file_name, 'r') as reader:
            item_list = json.loads(reader.read())
            for item in item_list:
                repo_to_total_count[repo] += item['total']
                username = item['author']['login']
                repo_to_contributor_data[repo][username] = item
    repo_to_pos_commiter = {}
    repo_to_neg_commiter = {}
    for repo, _ in repo_to_total_count.items():
        repo_to_pos_commiter[repo] = set()
        repo_to_neg_commiter[repo] = set()

    for record in records:
        repo = record.repo
        username = commit_to_username[int(record.id)]
        if username != 'None-':
            if record.label == 1:
                repo_to_pos_commiter[repo].add(username)
            if record.label == 0:
                repo_to_neg_commiter[repo].add(username)

    pos_commiter_percents = []
    neg_commiter_percents = []
    threshold = 0.05
    pos_threshold_count = 0
    neg_threshold_count = 0
    for repo, _ in repo_to_pos_commiter.items():
        if repo_to_total_count[repo] < 100:
            continue
        # print("{}       pos: {},    neg:  {}".format(repo, len(repo_to_pos_commiter[repo]), len(repo_to_neg_commiter[repo])))
        # print("Number of commits of positive commiters:")
        for username in repo_to_pos_commiter[repo]:
            if username in repo_to_contributor_data[repo]:
                item = repo_to_contributor_data[repo][username]
                total = item['total']
                percent = total / repo_to_total_count[repo]
                if percent < threshold:
                    pos_threshold_count += 1
                rounded_percent = round(percent * 100)
                # print("{}       =>      {}".format(total, percent))
                pos_commiter_percents.append(rounded_percent)
        # print("Number of commits of negative commiters:")
        for username in repo_to_neg_commiter[repo]:
            if username in repo_to_contributor_data[repo]:
                item = repo_to_contributor_data[repo][username]
                total = item['total']
                percent = total / repo_to_total_count[repo]
                if percent < threshold:
                    neg_threshold_count +=1
                rounded_percent = round(percent * 100)
                # print("{}       =>      {}".format(total, percent))
                neg_commiter_percents.append(rounded_percent)
    pos_commiter_percents = sorted(pos_commiter_percents)
    neg_commiter_percents = sorted(neg_commiter_percents)
    analyzer.plot_most_common_data(pos_commiter_percents, 30, "Commiter's commit percentage of positive records",
                                   "percent", "Number of committers", 10)
    analyzer.plot_most_common_data(neg_commiter_percents, 30,
                                   "Commiter's commit percentage of negative records",
                                   "percent", "Number of commiters", 100)
    print("Pos threshold percent: {}".format(pos_threshold_count/len(pos_commiter_percents)))
    print("Neg threshold percent: {}".format(neg_threshold_count/len(neg_commiter_percents)))


def get_previous_commit_time(date_unix, contributor_data):
    max_date = 0
    weeks_data = contributor_data['weeks']
    for commit_data in weeks_data:
        current_date = commit_data['w']
        has_commit = commit_data['c'] > 0
        if date_unix > current_date > max_date and has_commit:
            max_date = current_date

    return max_date


def get_next_commit_time(date_unix, contributor_data):
    min_date = 999999999999
    weeks_data = contributor_data['weeks']

    for commit_data in weeks_data:
        current_date = commit_data['w']
        has_commit = commit_data['c'] > 0
        if date_unix < current_date < min_date and has_commit:
            min_date = current_date

    return min_date


def get_num_commits_before(date_unix, contributor_data, seconds):
    date_unix_before = date_unix - seconds
    weeks_data = contributor_data['weeks']
    commit_count = 0
    for commit_data in weeks_data:
        commit_date = commit_data['w']
        if date_unix_before < commit_date < date_unix:
            commit_count += commit_data['c']

    return commit_count


def get_num_commits_after(date_unix, contributor_data, seconds):
    date_unix_after = date_unix + seconds
    weeks_data = contributor_data['weeks']
    commit_count = 0
    for commit_data in weeks_data:
        commit_date = commit_data['w']
        if date_unix < commit_date < date_unix_after:
            commit_count += commit_data['c']

    return commit_count


def analyze_q4():
    global max_date
    global min_date
    records = data_loader.load_records(labeled_record_file_path)
    id_to_record = {}

    commit_to_username = get_commit_username()

    for record in records:
        id_to_record[int(record.id)] = record

    repo_to_username = {}
    with open('../data/github_statistics/commit_username_clean.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            repo = row[1]
            if repo not in repo_to_username:
                repo_to_username[repo] = set()
            username = row[3]
            if username != 'None-':
                repo_to_username[repo].add(username)

    contributor_activity_file_path = os.path.join(directory,
                                                  '../data/github_statistics/contributor_activity')

    repo_to_total_count = {}
    repo_to_contributor_data = {}
    for file_name in os.listdir(contributor_activity_file_path):
        if not file_name.endswith('.json'):
            continue
        record_id = int(file_name.split('.')[0])
        repo = id_to_record[record_id].repo
        repo_to_total_count[repo] = 0
        repo_to_contributor_data[repo] = {}
        with open(contributor_activity_file_path + '/' + file_name, 'r') as reader:
            item_list = json.loads(reader.read())
            for item in item_list:
                repo_to_total_count[repo] += item['total']
                username = item['author']['login']
                repo_to_contributor_data[repo][username] = item

    day_to_previous_pos = []
    day_to_after_pos = []
    day_to_previous_neg = []
    day_to_after_neg = []
    one_month_before_pos = []
    one_month_after_pos = []
    one_month_before_neg = []
    one_month_after_neg = []
    for record in records:
        repo = record.repo
        commit = record.commit
        date_string = commit.created_date
        # e.g 'Sun, 14 Jul 2019 21:13:06 GMT'
        date = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S GMT")
        date_unix = mktime(date.timetuple())
        username = "None-"
        if int(record.id) in commit_to_username:
            username = commit_to_username[int(record.id)]
        if username != "None-":
            if username in repo_to_contributor_data[repo]:
                contributor_data = repo_to_contributor_data[repo][username]
                previous_commit_time = get_previous_commit_time(date_unix, contributor_data)
                if previous_commit_time != max_date:
                    day_to_previous = int((date_unix - previous_commit_time) / 3600)
                    if record.label == 1:
                        day_to_previous_pos.append(day_to_previous)
                    if record.label == 0:
                        day_to_previous_neg.append(day_to_previous)

                next_commit_time = get_next_commit_time(date_unix, contributor_data)
                if next_commit_time != min_date:
                    day_to_after = int((next_commit_time - date_unix) / 3600)
                    if record.label == 1:
                        day_to_after_pos.append(day_to_after)
                    if record.label == 0:
                        day_to_after_neg.append(day_to_after)

                num_commits_one_month_before = get_num_commits_before(date_unix, contributor_data, 3600 * 24 * 30)
                num_commits_one_month_after = get_num_commits_after(date_unix, contributor_data, 3600 * 24 * 30)
                num_commits_one_week_before = get_num_commits_before(date_unix, contributor_data, 3600 * 24 * 7)
                num_commits_one_week_after = get_num_commits_after(date_unix,contributor_data, 3600 * 24 * 7)
                if record.label == 1:
                    one_month_before_pos.append(num_commits_one_month_before)
                    one_month_after_pos.append(num_commits_one_month_after)
                if record.label == 0:
                    one_month_before_neg.append(num_commits_one_month_before)
                    one_month_after_neg.append(num_commits_one_month_after)
                # print("{}       {}          {}".format(previous_commit_time, date_unix, next_commit_time))
                # print("{}           =>      {}".format(num_commits_one_month_before, num_commits_one_month_after))
                # print("{}           =>      {}".format(num_commits_one_week_before, num_commits_one_week_after))
                # print(record.label)
                # print("----------------")

    analyzer.plot_most_common_data(day_to_previous_pos, 50, "Day to the previous commit of the same commiter for positive records", "Day", "Number of committer", 10)
    analyzer.plot_most_common_data(day_to_after_pos, 40, "Day to the following commit of the same commiter for  positive records", "Day", "Number of committer", 10)
    analyzer.plot_most_common_data(day_to_previous_neg, 50, "Day to the previous commit of the same commiter for negative records", "Day",
                                   "Number of committer", 10)
    analyzer.plot_most_common_data(day_to_after_neg, 50, "Day to the following commit of the same commiter for negative records", "Day",
                                   "Number of committer", 10)
    analyzer.plot_most_common_data(one_month_before_pos, 50,
                                   "Number of commits commiters made one month before for positive records", "Number of commits",
                                   "Number of committer", 10)
    analyzer.plot_most_common_data(one_month_after_pos, 50,
                                   "Number of commits commiters made one month after for positive records",
                                   "Number of commits",
                                   "Number of committer", 10)
    analyzer.plot_most_common_data(one_month_before_neg, 50,
                                   "Number of commits commiters made one month before for negative records",
                                   "Number of commits",
                                   "Number of committer", 50)
    analyzer.plot_most_common_data(one_month_after_neg, 50,
                                   "Number of commits commiters made one month after for negative records",
                                   "Number of commits",
                                   "Number of committer", 100)


def retrieve_github_file_history():
    token_index = 0
    access_token = access_tokens[token_index]
    records = data_loader.load_records(labeled_record_file_path)
    file_history_folder_path = os.path.join(directory, '../data/github_statistics/file_history')

    file_name_set = set()
    for file_name in os.listdir(file_history_folder_path):
        file_name_set.add(file_name)

    request_count = 0
    for record in records:
        record_id = record.id
        repo = utils.clear_github_prefix(record.repo)
        print("Repo: {}".format(repo))
        commit = record.commit

        for i in range(len(commit.files)):

            file_path = commit.files[i].file_name
            saved_file_name = str(record_id) + '_' + str(i) + '.json'
            if saved_file_name in file_name_set:
                continue
            saved_data_file_path = file_history_folder_path + '/' + str(record_id) + '_' + str(i) + '.json'
            print(file_path)
            headers = {'Authorization': 'bearer ' + access_token}
            response = requests.get(url='https://api.github.com/repos/' + repo + '/commits?path=' + file_path,
                                    headers=headers)
            request_count += 1
            if request_count % 1000 == 0:
                print("Delay for 1 minutes...")
                time.sleep(60*1)
                token_index += 1
                if token_index == 4:
                    token_index = 0
                access_token = access_tokens[token_index]

            data = response.json()

            with open(saved_data_file_path, mode='w') as file:
                file.write(json.dumps(data))


def retrieve_commit_date():
    week_unix = int('1367712000')
    time = datetime.utcfromtimestamp(week_unix).date()
    print()


def retrieve_github_assignee():
    records = data_loader.load_records(labeled_record_file_path)
    repo_to_id = {}
    assignee_folder_path = os.path.join(directory, '../data/github_statistics/assignee')
    for record in records:
        repo = record.repo
        repo_to_id[repo] = record.id

    for repo, id in repo_to_id.items():
        print(repo)
        repo = utils.clear_github_prefix(repo)
        assignee_file_path = assignee_folder_path + '/assignee_' + str(id) + ".json"
        headers = {'Authorization': 'bearer ' + access_tokens[0]}
        response = requests.get(url='https://api.github.com/repos/' + repo + '/assignees', headers=headers)
        data = response.json()

        with open(assignee_file_path, mode='w') as file:
            file.write(json.dumps(data))


def do_something():
    records = data_loader.load_records(labeled_record_file_path)

    for record in records:
       if int(record.id) == 7016:
           print(record.repo)


def analyze_assignee_info():
    repo_to_assignee = {}
    id_to_repo = {}
    records = data_loader.load_records(labeled_record_file_path)
    commit_to_username = get_commit_username()
    for record in records:
        id_to_repo[int(record.id)] = record.repo

    assignee_folder_path = os.path.join(directory, '../data/github_statistics/assignee')

    for file_name in os.listdir(assignee_folder_path):
        id = file_name.split('_')[1].split('.')[0]
        repo = id_to_repo[int(id)]
        repo_to_assignee[repo] = []
        with open(assignee_folder_path + '/' + file_name, 'r') as reader:
            content = reader.read()
            data_list = json.loads(content)
            for data in data_list:
                repo_to_assignee[repo].append(data['login'])

    pos_assignee = 0
    neg_assignee = 0
    pos_total = 0
    neg_total = 0
    for record in records:
        repo = record.repo
        record_id = int(record.id)
        if record_id not in commit_to_username:
            continue

        username = commit_to_username[record_id]
        if username == "None-":
            continue
        if repo in repo_to_assignee:
            if record.label == 1:
                pos_total += 1
                if username in repo_to_assignee[repo]:
                    pos_assignee += 1
            if record.label == 0:
                neg_total += 1
                if username in repo_to_assignee[repo]:
                    neg_assignee += 1

    print("{}       {}      {}".format(pos_assignee, pos_total, pos_assignee/pos_total))
    print("{}       {}      {}".format(neg_assignee, neg_total, neg_assignee / neg_total))
    print()


def retrieve_github_user():
    commit_to_username = get_commit_username()
    username_set = set()
    for x, y in commit_to_username.items():
        if y == "None-":
            continue
        username_set.add(y)

    user_folder_path = os.path.join(directory, '../data/github_statistics/user')
    for username in username_set:
        print(username)
        user_info_file_path = user_folder_path + '/' + username + ".json"
        headers = {'Authorization': 'bearer ' + access_tokens[0]}
        response = requests.get(url='https://api.github.com/users/' + username, headers=headers)
        data = response.json()
        with open(user_info_file_path, 'w') as file:
            file.write(json.dumps(data))


def analyze_user_info():
    commit_to_username = get_commit_username()
    records = data_loader.load_records(labeled_record_file_path)
    user_folder_path = os.path.join(directory, '../data/github_statistics/user')
    username_to_data = {}

    for file_name in os.listdir(user_folder_path):
        username = file_name.split(".")[0]
        with open(user_folder_path + '/' + file_name, 'r') as reader:
            data = json.loads(reader.read())
            username_to_data[username] = data

    pos_user_set = set()
    neg_user_set = set()
    for record in records:
        if int(record.id) not in commit_to_username:
            continue
        username = commit_to_username[int(record.id)]
        if username == "None-":
            continue
        if record.label == 1:
            pos_user_set.add(username)
        else:
            neg_user_set.add(username)

    pos_followers = []
    neg_followers = []
    pos_repos = []
    neg_repos = []
    for user in pos_user_set:
        pos_followers.append(username_to_data[user]['followers'])
        pos_repos.append(username_to_data[user]['public_repos'])

    for user in neg_user_set:
        neg_followers.append(username_to_data[user]['followers'])
        neg_repos.append(username_to_data[user]['public_repos'])

    pos_followers = pos_followers[:len(pos_followers) - 10]
    neg_followers = neg_followers[:len(neg_followers) - 10]
    pos_followers.sort()
    neg_followers.sort()
    pos_repos.sort()
    neg_repos.sort()

    plt.hist(x=pos_followers, bins=100)
    plt.title("pos followers")
    plt.xlabel("number")
    plt.ylabel("Counter")
    plt.show()

    plt.hist(x=neg_followers, bins=100)
    plt.title("neg followers")
    plt.xlabel("number")
    plt.ylabel("counter")
    plt.show()


    plt.boxplot([pos_followers, neg_followers])
    plt.title("Distribution in the number of followers of committers")
    plt.xticks([1, 2], ['Positive class', 'Negative class'])
    plt.ylabel("Number of followers")
    plt.show()

    plt.boxplot([pos_repos, neg_repos])
    plt.title("Distribution in the number of joined repositories of committers")
    plt.ylabel("Number of joined repositories")
    plt.xticks([1, 2], ['Positive class', 'Negative class'])
    plt.show()


    print(statistics.mean(pos_followers))
    print(statistics.median(pos_followers))
    print(statistics.stdev(pos_followers))
    print("---------------")
    print(statistics.mean(neg_followers))
    print(statistics.median(neg_followers))
    print(statistics.stdev(neg_followers))
    print("---------------")
    print(statistics.mean(pos_repos))
    print(statistics.median(pos_repos))
    print(statistics.stdev(pos_repos))
    print("---------------")
    print(statistics.mean(neg_repos))
    print(statistics.median(neg_repos))
    print(statistics.stdev(neg_repos))
    print("---------------")


def retrieve_pull_request():
    records = data_loader.load_records(labeled_record_file_path)
    token_index = 0

    repo_to_commit_id = {}
    commit_id_to_repo = {}
    for record in records:
        record_id = int(record.id)
        repo = record.repo
        repo_to_commit_id[repo] = record_id
        commit_id_to_repo[record_id] = repo

    folder_path = os.path.join(directory, "../data/github_statistics/pull_request")
    processed_project = set()
    for file_name in os.listdir(folder_path):
        commit_id = int(file_name.split('.')[0])
        processed_project.add(commit_id_to_repo[commit_id])

    for repo_name, commit_id in repo_to_commit_id.items():
        print(repo_name)
        if commit_id in [1738, 2543, 3107, 3649, 3898, 5238, 5562, 7160]:
            continue
        if utils.clear_github_prefix(repo_name) in ['apache/hadoop', 'elastic/elasticsearch']:
            continue
        if repo_name in processed_project:
            continue

        gh = github.Github(login_or_token=access_tokens[token_index], per_page=100)
        repo = gh.get_repo(utils.clear_github_prefix(repo_name))
        pulls = repo.get_pulls(state='all')
        total = pulls.totalCount
        total_page = int(total / 100)
        pull_request_file_path = os.path.join(directory, "../data/github_statistics/pull_request/" + str(commit_id) + ".csv")
        rows = []
        request_count = 0
        for i in range(total_page + 1):
            for pull in pulls.get_page(i):
                print(pull.url)
                request_count += 1
                user_login = "None-"
                assignee_login = "None-"
                merged_by_login = "None-"

                if pull.user is not None and hasattr(pull.user, "login") and pull.user.login is not None:
                    user_login = pull.user.login

                if pull.assignee is not None and hasattr(pull.assignee, "login") and pull.assignee.login is not None:
                    assignee_login = pull.assignee.login

                if pull.merged_by is not None and hasattr(pull.merged_by, "login") and pull.merged_by.login is not None:
                    merged_by_login = pull.merged_by.login

                created_at_timestamp = -1
                merged_at_timestamp = -1
                if pull.created_at is not None:
                    created_at_timestamp = mktime(pull.created_at.timetuple())
                if pull.merged_at is not None:
                    merged_at_timestamp = mktime(pull.merged_at.timetuple())

                rows.append([pull.id, pull.number, pull.url, created_at_timestamp, pull.state, user_login,
                             assignee_login, merged_by_login, merged_at_timestamp, pull.commits, pull.changed_files,
                             pull.additions, pull.deletions])

            if request_count > 1000:
                request_count = 0
                token_index += 1
                if token_index == 4:
                    token_index = 0
                gh = github.Github(login_or_token=access_tokens[token_index], per_page=100)
                repo = gh.get_repo(utils.clear_github_prefix(repo_name))
                pulls = repo.get_pulls('all')

        print("Writing to file...")
        with open(pull_request_file_path, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'number', 'url', 'created_at', 'state', 'user_login', 'assignee_login',
                             'merged_by_login', 'merged_at', 'commits', 'changed_files', 'additions', 'deletions'])
            for row in rows:
                writer.writerow(row)


def retrieve_pull_request_fast():
    records = data_loader.load_records(labeled_record_file_path)
    token_index = 0

    repo_to_open_pr = {}
    repo_to_close_pr = {}
    repo_to_commit_id = {}
    repo_to_username = deep_experiment.get_repo_to_username()

    for record in records:
        record_id = int(record.id)
        repo = record.repo
        repo_to_commit_id[repo] = record_id

    for repo_name, commit_id in repo_to_commit_id.items():
        print(repo_name)
        gh = github.Github(login_or_token=access_tokens[token_index], per_page=100)
        repo = gh.get_repo(utils.clear_github_prefix(repo_name))
        pulls = repo.get_pulls(state='open')
        print(pulls.totalCount)
        pulls = repo.get_pulls(state='closed')
        print(pulls.totalCount)
        if repo not in repo_to_username:
            continue
        for username in repo_to_username[repo]:
            pulls = repo.get_pulls(state='open', head='')


def collect_file_history_statistics():
    file_history_folder_path = os.path.join(directory, '../data/github_statistics/file_history')
    file_statistics_folder_path = os.path.join(directory, '../data/github_statistics/file_history_statistics')
    token_index = 0
    request_count = 0
    access_token = access_tokens[token_index]
    file_count = 0

    record_to_files = {}

    records = data_loader.load_records(labeled_record_file_path)
    for record in records:
        record_to_files[int(record.id)] = []
        for file in record.commit.files:
            record_to_files[int(record.id)].append(file.file_name)

    processed_file = set()
    for file_name in os.listdir(file_statistics_folder_path):
        processed_file.add(file_name)

    for file_name in os.listdir(file_history_folder_path):
        file_count += 1
        print("{}       {}".format(file_name, file_count))
        file_path = file_history_folder_path + '/' + file_name

        if file_name in processed_file:
            continue

        parts = file_name.split('.')[0].split('_')
        record_id = int(parts[0])
        file_index = int(parts[1])
        commit_file_name = record_to_files[record_id][file_index]
        if 'src/test' in commit_file_name:
            continue
        if not commit_file_name.endswith(('.java', '.c', '.h')):
            continue

        with open(file_path, 'r') as file:
            data_list = json.loads(file.read())

        file_statistics_data = []

        for data in data_list:
            request_count += 1
            commit_api_url = data['url']
            headers = {'Authorization': 'bearer ' + access_token}
            response = requests.get(url=commit_api_url, headers=headers)
            json_data = response.json()
            if 'files' in json_data:
                for file_data in json_data['files']:
                    if 'patch' in file_data:
                        file_data.pop('patch')
                file_statistics_data.append(json_data)

            if request_count == 3000:
                request_count = 0
                token_index = (token_index + 1) % 4
                access_token = access_tokens[token_index]

        with open(file_statistics_folder_path + '/' + file_name, 'w') as writer:
            writer.write(json.dumps(file_statistics_data))


def collect_file_history_statistics_fast():
    file_history_folder_path = os.path.join(directory, '../data/github_statistics/file_history')
    file_statistics_folder_path = os.path.join(directory, '../data/github_statistics/history_fast')
    token_index = 0

    record_to_repo = {}
    repo_to_record = {}
    repo_to_commit = {}
    id_to_record = {}

    entity_encoder = EntityEncoder()

    records = data_loader.load_records(labeled_record_file_path)
    for record in records:
        repo = record.repo
        record_id = int(record.id)
        record_to_repo[record_id] = repo
        id_to_record[record_id] = record
        repo_to_record[repo] = record_id

    processed_repo = set()
    for file_name in os.listdir(file_statistics_folder_path):
        parts = file_name.split('.')[0].split('_')
        record_id = int(parts[0])
        repo = record_to_repo[record_id]
        processed_repo.add(repo)

    for file_name in os.listdir(file_history_folder_path):
        file_path = file_history_folder_path + '/' + file_name
        parts = file_name.split('.')[0].split('_')
        record_id = int(parts[0])
        repo = record_to_repo[record_id]
        # print(repo)
        if repo in processed_repo:
            continue

        if repo not in repo_to_commit:
            repo_to_commit[repo] = set()
        with open(file_path, 'r') as reader:
            data_list = json.loads(reader.read())

        for data in data_list:
            try:
                sha = data['sha']
                repo_to_commit[repo].add(sha)
            except TypeError:
                pass
                # print()

    total_commit = 0
    for repo_name, sha_list in repo_to_commit.items():
        total_commit += len(sha_list)

    request_count = 0
    current_count = 0
    gh = github.Github(access_tokens[token_index])
    for repo_name, sha_list in repo_to_commit.items():
        repo = gh.get_repo(utils.clear_github_prefix(repo_name))
        commit_list = []
        for sha in sha_list:
            current_count += 1
            print('{}/{}        =>      {}'.format(current_count, total_commit, sha))
            request_count += 1
            commit = repo.get_commit(sha)
            commit_files = []
            for file in commit.files:
                commit_file = GithubCommitFile(file_name=file.filename, patch=None, status=file.status,
                                               additions=file.additions, deletions=file.deletions, changes=file.changes)
                commit_files.append(commit_file)
            try:
                committer_login = "None-"
                if hasattr(commit.committer, 'login'):
                    committer_login = commit.committer.login

                github_commit = GithubCommit(author_name=committer_login,
                                             created_date=commit.last_modified, files=commit_files)
            except Exception:
                continue
            print(github_commit)
            commit_list.append(github_commit)
            if request_count == 1000:
                request_count = 0
                token_index = (token_index + 1) % 4
                gh = github.Github(access_tokens[token_index])
                repo = gh.get_repo(utils.clear_github_prefix(repo_name))

        repo_id = repo_to_record[repo_name]
        print("Writting to file...")
        with open(file_statistics_folder_path + '/' + str(repo_id) + '.json', 'w') as writer:
            writer.write(entity_encoder.encode(commit_list))


collect_file_history_statistics_fast()
# collect_file_history_statistics()