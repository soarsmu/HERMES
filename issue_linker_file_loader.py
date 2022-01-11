import utils
import os


directory = os.path.dirname(os.path.abspath(__file__))
jira_ticket_file_path = os.path.join(directory, 'data/jira_issue_batch_data')


def get_relevant_files():
    apache_key_set = set()
    lines = utils.read_lines('repo_to_apache_key.txt')
    for line in lines:
        parts = line.split("\t\t")
        repo = parts[0]
        keys = parts[1].split(',')
        for key in keys:
            apache_key_set.add(key)

    apache_keys = tuple(apache_key_set)

    count = 0
    for file_name in os.listdir(jira_ticket_file_path):
        if file_name.startswith(apache_keys):
            print(file_name)
            count += 1

    print(count)


get_relevant_files()