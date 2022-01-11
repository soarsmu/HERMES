import github
import utils
import urllib3
from entities import GithubCommit, GithubCommitFile, EntityEncoder
from utils import print_line_seperator


urllib3.disable_warnings()
# gh = github.Github("2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24")
gh = github.Github("dd28467db1a1c81d7dcef7c17ae8bc0d850ef9cf")
commit_data_file_path = "../data/github_commit/"

entity_encoder = EntityEncoder()


def process_commit_files(files):
    commit_files = []
    for file in files:
        commit_file = GithubCommitFile(file_name=file.filename, patch=file.patch, status=file.status,
                                       additions=file.additions, deletions=file.deletions, changes=file.changes)
        commit_files.append(commit_file)
    return commit_files


def write_commit_data_to_json(github_commit, record_id):
    data_json = entity_encoder.encode(github_commit)
    file_path = commit_data_file_path + record_id + ".txt"
    with open(file_path, 'w') as text_file:
        text_file.write(data_json)
    # github_commit = GithubCommit(json_value=data_json)
    # for commit_file in github_commit.files:
    #     print_line_seperator()


def process_commit(commit, record_id):
    commit_files = process_commit_files(commit.files)
    github_commit = GithubCommit(author_name=commit.commit.author.name,
                                 created_date=commit.last_modified, files=commit_files)
    write_commit_data_to_json(github_commit, record_id)


def records_to_repo_to_record_map(records):
    repo_to_record = {}
    for record in records:
        repo_name = record.repo
        if repo_name not in repo_to_record:
            repo_to_record[repo_name] = []
        repo_to_record[repo_name].append(record)

    return repo_to_record


def extract_commit():
    file_name = '../MSR2019/experiment/full_dataset.csv'
    records = utils.extract_record(file_name, has_message=True)
    repo_to_record = records_to_repo_to_record_map(records)

    extracted_commits = utils.read_lines('../extracted_commits.txt')
    for repo_name, records in repo_to_record.items():
        try:
            print(repo_name)
            repo = gh.get_repo(utils.clear_github_prefix(repo_name))
            for record in records:
                if record.commit_id in extracted_commits:
                    continue
                commit = repo.get_commit(record.commit_id)
                if utils.is_not_large_commit(commit):
                    process_commit(commit, record.id)
                extracted_commits.append(record.commit_id)
                print(record.commit_id)
        except github.RateLimitExceededException:
            print('Rate limit reached')
            utils.write_lines(extracted_commits, '../extracted_commits.txt')
            break
