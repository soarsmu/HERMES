import utils
import github_issue_extractor
import jira_ticket_extractor


def extract_record(filename):
    records = utils.extract_record(file_name=file_name, has_message=True)
    return records


def do_some_analysis(records):
    repo_set = set()
    for record in records:
        repo_set.add(record.repo)
    print("repo count: {}".format(len(repo_set)))

file_name = '../MSR2019/dataset/dataset_with_message_fixed.csv'
records = extract_record(file_name)
# print(len(records))
# github_record_set = github_extractor.process_github()
# jira_record_set = jira_extractor.process_jira_data()
# enhanced_record_count = len(github_record_set.union(jira_record_set))
# print("Number of enhanced records: {}".format(enhanced_record_count))

