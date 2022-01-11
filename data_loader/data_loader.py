import utils
import os
from entities import *

full_data_with_features_file_path = '../MSR2019/experiment/full_dataset_with_all_features.txt'
record_file_path = '../MSR2019/experiment/full_dataset_with_patches.txt'
github_commit_file_path = '../data/github_commit'
github_issue_file_path = '../data/github_issue'
jira_ticket_file_path = '../data/jira_ticket'
record_full_file_path = '../data/record_full.txt'


def load_github_commit(id_to_record):
    record_id_list = []
    # some commit with very large patches need to be removed
    filtered_records = []
    for file_name in os.listdir(github_commit_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        record_id_list.append(record_id)

    print(len(record_id_list))

    for file_name in os.listdir(github_commit_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        with open(github_commit_file_path + '/' + file_name) as file:
            content = file.read()
            commit = GithubCommit(json_value=content)
            record = id_to_record[record_id]
            record.commit = commit

    for id in record_id_list:
        filtered_records.append(id_to_record[id])

    return filtered_records


def load_github_issue(id_to_record):
    for file_name in os.listdir(github_issue_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        with open(github_issue_file_path + '/' + file_name) as file:
            json_raw = file.read()
            json_dict_list = json.loads(json_raw)
            github_issues = []
            for json_dict in json_dict_list:
                github_issues.append(GithubIssue(json_value=json_dict))
        if record_id in id_to_record:
            id_to_record[record_id].github_issue_list = github_issues


def load_jira_ticket(id_to_record):
    for file_name in os.listdir(jira_ticket_file_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        with open(jira_ticket_file_path + '/' + file_name) as file:
            json_raw = file.read()
            json_dict_list = json.loads(json_raw)
            jira_tickets = []
            for json_dict in json_dict_list:
                jira_tickets.append(JiraTicket(json_value=json_dict))
        id_to_record[record_id].jira_ticket_list = jira_tickets


def write_full_data_to_file(file_path):
    print("Writing full records to file...")
    records = load_records(record_file_path)
    
    id_to_record = {}
    for record in records:
        if utils.is_not_large_commit(record.commit):
            id_to_record[record.id] = record

    # records = load_github_commit(id_to_record)
    load_github_issue(id_to_record)
    load_jira_ticket(id_to_record)

    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(records)

    with open(file_path, 'w') as file:
        file.write(json_value)
    print("Finishing writing")


def load_records(file_path):
    print("Start loading records...")

    records = []
    with open(file_path, 'r') as file:
        json_raw = file.read()
        json_dict_list = json.loads(json_raw)
        for json_dict in json_dict_list:
            records.append(Record(json_value=json.dumps(json_dict)))

    print("Finish loading records")

    for record in records:
        if record.label is not None:
            if record.label == 'pos':
                record.label = 1
            elif record.label == 'neg':
                record.label = 0

    return records


# write_full_data_to_file(full_data_with_features_file_path)

# def retrieve_records_with_limited_features():
#     record_file_path = "/Users/nguyentruonggiang/Desktop/SMU/project/AutoVulCuration/MSR2019/experiment/full_dataset_with_all_features.txt"
#     records = load_records(record_file_path)
#     for record in records:
#         record.jira_ticket_list = None
#         record.github_issue_list = None
#         record.issue_info = None
#         record.code_terms = None
#         record.text_terms_parts = None
#         record.__delattr__("jira_ticket_list")
#         record.__delattr__("github_issue_list")
#         record.__delattr__("issue_info")
#         record.__delattr__("code_terms")
#         record.__delattr__("text_terms_parts")
#     entity_encoder = EntityEncoder()
#     json_value = entity_encoder.encode(records)
#
#     print("Writing records...")
#     with open('/Users/nguyentruonggiang/Desktop/SMU/project/AutoVulCuration/MSR2019/experiment/dataset.json', 'w') as file:
#         file.write(json_value)
#     print("Finishing writing")
#     print(len(records))

