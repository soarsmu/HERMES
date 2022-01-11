import urllib3
import github
import random
import utils
from entities import Record, EntityEncoder, GithubCommit
import os
from data_loader import data_loader
from data_loader import commit_extractor
import math

urllib3.disable_warnings()
gh = github.Github("2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24")
# gh = github.Github("dd28467db1a1c81d7dcef7c17ae8bc0d850ef9cf")
number_of_commit_per_repo = 200

directory = os.path.dirname(os.path.abspath(__file__))
labeled_record_file_path = os.path.join(directory, '../MSR2019/experiment/full_dataset_with_all_features.txt')
unlabeled_folder_path = os.path.join(directory, '../data/unlabeled')

excluded_repos = ['https://github.com/apache/sling', 'https://github.com/djmdjm/jBCrypt']


def retrieve_unlabeled_data(repo_name, current_commit_id_list):
    repo = gh.get_repo(utils.clear_github_prefix(repo_name))
    commits = repo.get_commits()
    total_commit = commits.totalCount

    item_per_page = 30

    retrieved_count = 0
    retrieved_commit_id_list = []
    retrieved_records = []

    # if after number_of_try but can not find commit, return
    number_of_try = 0
    max_try = 10
    while retrieved_count < number_of_commit_per_repo:
        random_commit_id = random.randrange(total_commit)
        page_id = math.floor(random_commit_id / item_per_page)
        if page_id == 0:
            continue
        offset = random_commit_id % item_per_page
        page = commits.get_page(page_id)
        commit = page[offset]

        url = repo_name + '/commit/' + commit.sha
        if commit.sha not in current_commit_id_list \
                and commit.sha not in retrieved_commit_id_list \
                and utils.is_not_large_commit(commit)\
                and utils.contain_java_file(commit):
            retrieved_commit_id_list.append(commit.sha)
            commit_files = commit_extractor.process_commit_files(commit.files)
            github_commit = GithubCommit(author_name=commit.commit.author.name,
                                         created_date=commit.last_modified, files=commit_files)
            record = Record(id=-1, repo=repo_name, commit_id=commit.sha,
                                         commit_message=commit.commit.message)
            record.commit = github_commit
            retrieved_records.append(record)

            retrieved_count += 1
            print(url)
            number_of_try = 0
        else:
            number_of_try += 1
        if number_of_try > max_try:
            break

    return retrieved_records


def load_labeled_and_unlabeled_records(unlabeled_size):
    labeled_records =data_loader.load_records(os.path.join(directory, labeled_record_file_path))

    unlabeled_records = []

    unlabeled_file_count = 0
    for file_name in os.listdir(unlabeled_folder_path):
        if file_name.endswith('.txt'):
            unlabeled_data = data_loader.load_records(unlabeled_folder_path + '/' + file_name)
            unlabeled_records.extend(unlabeled_data)
            unlabeled_file_count += 1
        if unlabeled_size != -1 and len(unlabeled_records) > unlabeled_size:
            break
    return labeled_records, unlabeled_records, unlabeled_file_count


def retrieve_data(records, unlabeled_file_count):
    generated_projects = utils.read_lines('../generated_projects.txt')

    repo_to_commits = {}
    for record in records:
        repo = record.repo
        if repo not in repo_to_commits:
            repo_to_commits[repo] = []
        repo_to_commits[repo].append(record.commit_id)

    retrieved_records = []
    try:
        for repo, commit_ids in repo_to_commits.items():
            print(repo)
            if repo not in generated_projects:
                # this project has only 4 commits => ignore it
                if repo in excluded_repos:
                    continue
                retrieved_records.extend(retrieve_unlabeled_data(repo, commit_ids))
                generated_projects.append(repo)
    except github.RateLimitExceededException:
        print("Limit reached, writing to file")
        write_retrieved_unlabeled_data_to_files(unlabeled_file_count, generated_projects, retrieved_records)
        return

    write_retrieved_unlabeled_data_to_files(unlabeled_file_count, generated_projects, retrieved_records)


def write_retrieved_unlabeled_data_to_files(unlabeled_file_count, generated_projects, retrieved_records):
    unlabeled_file_count += 1
    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(retrieved_records)

    print("Writing records...")
    with open(unlabeled_folder_path + '/unlabeled_data_' + str(unlabeled_file_count) + '.txt', 'w') as file:
        file.write(json_value)
    print("Finishing writing")

    utils.write_lines(generated_projects, '../generated_projects.txt')


def do_collect():
    labeled_records, unlabeled_records, unlabeled_file_count = load_labeled_and_unlabeled_records()
    records = []
    for record in labeled_records:
        records.append(record)
    for record in unlabeled_records:
        records.append(record)
    retrieve_data(records, unlabeled_file_count)

# load_labeled_and_unlabeled_records()