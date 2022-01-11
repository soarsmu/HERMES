import urllib3
import github
import random
import utils
from entities import Record
import csv

urllib3.disable_warnings()
# gh = github.Github("2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24")
gh = github.Github("dd28467db1a1c81d7dcef7c17ae8bc0d850ef9cf")
number_of_random_per_commit = 5





def get_random_commit(repo_name, positive_commit_id_list):
    repo = gh.get_repo(utils.clear_github_prefix(repo_name))
    commits = repo.get_commits()
    total_commit = commits.totalCount

    item_per_page = 30
    max_page = int(total_commit / item_per_page)

    random_count = 0
    random_commit_id_list = []
    random_records = []

    while random_count < number_of_random_per_commit * len(positive_commit_id_list):
        random_page = random.randrange(max_page + 1)
        page = commits.get_page(random_page)
        if len(page) == 0:
            continue
        commit = page[random.randrange(len(page))]
        url = repo_name + '/commit/' + commit.sha
        print(url)
        if commit.sha not in positive_commit_id_list \
                and commit.sha not in random_commit_id_list \
                and utils.is_not_empty_commit(commit) \
                and utils.is_not_large_commit(commit)\
                and utils.contain_java_file(commit):
            random_commit_id_list.append(commit.sha)
            print("Accepted")
            random_records.append(Record(id=-1, repo=repo_name, commit_id=commit.sha,
                                         commit_message=commit.commit.processed_message, label='neg'))
            random_count += 1
    return random_records


def write_negative_sample_to_csv(records):
    generated_projects = utils.read_lines('../generated_projects.txt')

    repo_to_commits = {}
    for record in records:
        repo = record.repo
        if repo not in repo_to_commits:
            repo_to_commits[repo] = []
        repo_to_commits[repo].append(record.commit_id)

    random_records = []
    try:
        for repo, positive_commits in repo_to_commits.items():
            print(repo)
            if repo not in generated_projects:
                # this project has only 4 commits => ignore it
                if repo == 'https://github.com/apache/sling':
                    continue
                random_records.extend(get_random_commit(repo, positive_commits))
                generated_projects.append(repo)
    except github.RateLimitExceededException:
        print("Limit reached, writing to file")
        with open('MSR2019/experiment/negative_samples_5.csv', mode='w') as csv_file:
            writer = csv.writer(csv_file)
            for record in random_records:
                writer.writerow([record.id, record.repo, record.commit_id, record.commit_message, record.label])
        utils.write_lines(generated_projects, '../generated_projects.txt')
        return

    with open('MSR2019/experiment/negative_samples_5.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        for record in random_records:
            writer.writerow([record.id, record.repo, record.commit_id, record.commit_message, record.label])
        utils.write_lines(generated_projects, '../generated_projects.txt')

# https://github.com/apache/sling => this project has only 4 commits?
# it moved to https://github.com/apache/sling-old-svn-mirror


# dataset_file_path = 'MSR2019/experiment/dataset_with_message.csv'
# records = utils.extract_record(dataset_file_path)
# write_negative_sample_to_csv(records)

# negative_samples_file_paths = ['MSR2019/experiment/negative_samples_1.csv',
#                                'MSR2019/experiment/negative_samples_2.csv',
#                                'MSR2019/experiment/negative_samples_3.csv',
#                                'MSR2019/experiment/negative_samples_4.csv',
#                                'MSR2019/experiment/negative_samples_5.csv']
#
# content = ''
#
# for file_path in negative_samples_file_paths:
#     with open(file_path, 'r') as file:
#         content += file.read()
#
# full_negative_file_path = 'MSR2019/experiment/full_negative.csv'
# with open(full_negative_file_path, 'w') as file:
#     file.write(content)

# positive_file_path = 'MSR2019/experiment/dataset_with_message.csv'
# negative_file_path = 'MSR2019/experiment/full_negative.csv'
# full_dataset_file_path = 'MSR2019/experiment/full_dataset.csv'
#
# positive_records = utils.extract_record(positive_file_path, has_message=True)
# negative_records = utils.extract_record(negative_file_path, has_message=True)
# start_index = int(positive_records[len(positive_records) - 1].id) + 1
# for record in negative_records:
#     record.id = str(start_index)
#     start_index += 1
#
# positive_records.extend(negative_records)
# print(len(positive_records))

