import github
import utils
from utils import print_line_seperator
import re
import urllib3
from entities import GithubIssue, GithubIssueComment, EntityEncoder
import json
import os

urllib3.disable_warnings()
gh = github.Github("2d99eeb58e5a01ac5bdcffa4af4c0c19717dee24")

invalid_issue_dictionary = {'19': ['#143893217'],
                            '24': ['#56164'],
                            '29': ['#2454'],
                            '30': ['#52372'],
                            '60': ['#100043648'],
                            '95': ['#158846330'],
                            '149': ['#138677887'],
                            '153': ['#145624295'],
                            '154': ['#145313231'],
                            '175': ['#54764'],
                            '201': ['#149701173'],
                            '218': ['#143846565'],
                            '230': ['#149701173'],
                            '236': ['#6630'],
                            '252': ['#54764', '#56164'],
                            '289': ['#158222161'],
                            '308': ['#1412'],
                            '318': ['#157376547'],
                            '336': ['#54682'],
                            '361': ['#54764'],
                            '382': ['#143844261'],
                            '395': ['#121682941'],
                            '402': ['#152093534', '#152164929'],
                            '417': ['#54764', '#56164'],
                            '421': ['#57031'],
                            '428': ['#129374221'],
                            '474': ['#154182646'],
                            '531': ['#158222161'],
                            '536': ['#157069100'],
                            '598': ['#315668'],
                            '620': ['#2097'],
                            '631': ['#129374221'],
                            '635': ['#56164'],
                            '729': ['#157377568'],
                            '742': ['#92065804'],
                            '764': ['#153420213'],
                            '769': ['#56814', '#56164'],
                            '778': ['#152093534', '#152164930'],
                            '850': ['#152093534', '#152164926'],
                            '959': ['#3637'],
                            '1010': ['#43327'],
                            '1107': ['#142550849'],
                            '1108': ['#140580003'],
                            '1360': ['#148875935'],
                            '1376': ['#66229608'],
                            '1385': ['#157486003'],
                            '1386': ['#72731888'],
                            '1389': ['#134026441'],
                            '1390': ['#108258400'],
                            '1392': ['#36234333'],
                            '1394': ['#100043648'],
                            '1397': ['#157060251'],
                            '1399': ['#157060251'],
                            '1400': ['#109738682'],
                            '1401': ['#104468594'],
                            '1404': ['#72924388'],
                            '1406': ['#112688633'],
                            '1407': ['#153568058'],
                            '1408': ['#43611117'],
                            '1410': ['#145000637'],
                            '1412': ['#119322315'],
                            '1413': ['#82182984'],
                            '1415': ['#74160792'],
                            '1416': ['#143817207'],
                            '1492': ['#62831']}


entity_encoder = EntityEncoder()
github_issue_data_file_path = "../data/github_issue/"


def write_github_issues_to_file(github_issue_list, record_id):
    file_path = github_issue_data_file_path + record_id + ".txt"
    json_dict_list = []
    for github_issue in github_issue_list:
        json_dict_list.append(entity_encoder.encode(github_issue))

    # this is example for deserialization
    # github_issue_new = []
    # for json_dict in json_dict_list:
    #     github_issue_new.append(GithubIssue(json_value=json_dict))

    with open(file_path, 'w') as text_file:
        text_file.write(json.dumps(json_dict_list))


def extract_github_issues(record):
    has_issue = False
    github_issue_regex = '#[0-9]{1,20}'
    matched_list = re.findall(github_issue_regex, record.commit_message)
    github_issue_list = []
    for issue_number in matched_list:
        if record.id not in invalid_issue_dictionary \
                or issue_number not in invalid_issue_dictionary[record.id]:
            github_issue_list.append(extract_issue_content(record, issue_number))
            has_issue = True
    return has_issue, github_issue_list


def extract_issue_content(record, issue_number):
    issue_number = issue_number[1:]
    repo_name = utils.clear_github_prefix(record.repo)
    repo = gh.get_repo(repo_name)
    issue = repo.get_issue(int(issue_number))
    title = issue.title
    body = issue.body
    author_name = issue.user.name
    # Todo currently using string value for datetime, check late
    created_at = str(issue.created_at)
    closed_at = str(issue.closed_at)
    closed_by = None
    if hasattr(issue, 'close_by'):
        closed_by = issue.closed_by.name
    last_modified = issue.last_modified

    original_comments = issue.get_comments(issue.created_at)
    extracted_comments = []
    for original_comment in original_comments:
        comment_body = original_comment.body
        comment_created_at = str(original_comment.created_at)
        comment_created_by = original_comment.user.name
        comment_last_modified = original_comment.last_modified
        extracted_comments.append(GithubIssueComment(body=comment_body, created_at=comment_created_at,
                                                     created_by=comment_created_by,
                                                     last_modified=comment_last_modified))
    github_issue = GithubIssue(title=title, body=body, author_name=author_name, created_at=created_at,
                               closed_at=closed_at, closed_by=closed_by,
                               last_modified=last_modified, comments=extracted_comments)
    return github_issue


def get_all_process_records(folder_path):
    process_record_ids = []
    for file_name in os.listdir(folder_path):
        record_id = file_name[:(len(file_name) - len('.txt'))]
        process_record_ids.append(record_id)
    return process_record_ids


def process_github():
    print('Processing github...')
    file_name = '../MSR2019/experiment/full_dataset.csv'
    records = utils.extract_record(file_name, has_message=True)
    record_count = 0
    record_set = set()
    processed_record_ids = get_all_process_records(github_issue_data_file_path)

    for record in records:
        print(record)

        if record.id in processed_record_ids:
            continue

        try:
            has_valid_issue, github_issue_list = extract_github_issues(record)
            if has_valid_issue:
                record_count += 1
                record_set.add(record)
                write_github_issues_to_file(github_issue_list, record.id)
        except github.UnknownObjectException:
            print("Error...")
    print("Number of records have valid issue: {}".format(record_count))
    print_line_seperator()
    return record_set


process_github()