from jira import JIRA, JIRAError
from entities import JiraTicket, JiraTicketComment, EntityEncoder
import json
import traceback
import utils
import os
from datetime import datetime
from data_loader import data_loader


# The directory that 'test.py' is stored
directory = os.path.dirname(os.path.abspath(__file__))

entity_encoder = EntityEncoder()
jira_issue_batch_folder_path = os.path.join(directory, "data/jira_issue_batch/")


def get_jira_ticket_info(issue):
    fields = issue.fields
    year_str = fields.created[:4]
    created_time = datetime.strptime(year_str, '%Y')
    if created_time.year < 2015:
        return None
    name = issue.key
    print(name)
    summary = fields.summary
    description = fields.description
    created_at = fields.created

    creator = ''
    if hasattr(fields, 'creator') and hasattr(fields.creator, 'displayName'):
        creator = fields.creator.displayName

    comments = []

    if not hasattr(fields, 'comment'):
        print("Issue has no comment")
    else:
        for ticket_comment in fields.comment.comments:
            body = ticket_comment.body
            created_at = ticket_comment.created
            updated_at = ticket_comment.updated
            comments.append(JiraTicketComment(created_by=None, body=body,
                                              created_at=created_at, updated_at=updated_at))

    jira_ticket = JiraTicket(name=name, summary=summary, description=description,
                             created_at=created_at, creator=creator, comments=comments)

    return jira_ticket


def write_ticket_list_to_file(jira_ticket_list, file_path):
    json_dict_list = []
    for jira_ticket in jira_ticket_list:
        json_dict_list.append(entity_encoder.encode(jira_ticket))

    with open(file_path, 'w') as text_file:
        text_file.write(json.dumps(json_dict_list))


def list_all_issues(project_key, jira):
    block_size = 100
    block_num = 0
    jql = "project=" + project_key
    stop = False
    while True:
        if stop:
            break
        start_idx = block_num * block_size
        try:
            issues = jira.search_issues(jql, start_idx, block_size,
                                        fields='comment,summary,description,creator,created')
            if len(issues) == 0:
                # Retrieve issues until there are no more to come
                break
            block_num += 1
            jira_ticket_list = []
            for issue in issues:
                jira_ticket = get_jira_ticket_info(issue)
                if jira_ticket is None:
                    stop = True
                else:
                    jira_ticket_list.append(jira_ticket)
            jira_ticket_list = [get_jira_ticket_info(issue) for issue in issues]
            file_path = jira_issue_batch_folder_path + project_key + '-' + str(block_num) + '.txt'
            write_ticket_list_to_file(jira_ticket_list, file_path)
            print("Block num: {}".format(block_num))
        except (JIRAError, Exception) as e:
            traceback.print_exc()
            return


def crawl_project_issue(project, jira):
    print("Processing project: {}".format(project.name))
    project_key = project.key
    list_all_issues(project_key, jira)


def get_ticket_prefix_set():
    reserved_keyword = ['INT']
    ticket_prefix_set = set()
    records = data_loader.load_records(os.path.join(directory, 'MSR2019/experiment/full_dataset_with_all_features.txt'))
    for record in records:
        if len(record.jira_ticket_list) > 0:
            for ticket in record.jira_ticket_list:
                prefix = ticket.name.split('-')[0]
                if prefix not in reserved_keyword:
                    ticket_prefix_set.add(ticket.name.split('-')[0])

    return ticket_prefix_set


def crawl_jira_issue():
    jira_server_list = ['https://issues.jenkins.io',
                        'https://issues.apache.org/jira',
                        'https://jira.spring.io',
                        'https://issues.redhat.com']

    red_hat_username = 'yourusername'
    red_hat_password = 'yourpassword'

    project_key_set = get_ticket_prefix_set()

    stop = False
    for jira_server in jira_server_list:
        if stop:
            break

        if jira_server == 'https://issues.redhat.com':
            jira = JIRA(server=jira_server, basic_auth=(red_hat_username, red_hat_password))
        else:
            jira = JIRA(server=jira_server)

        project_list = jira.projects()
        indices_file_path = os.path.join(directory, 'processed_projects.txt')
        processed_project = utils.read_lines(indices_file_path)
        for project in project_list:
            if stop:
                break

            if project.key not in project_key_set:
                continue

            code = jira_server + '\t\t' + project.key
            print(code)
            if code not in processed_project:
                try:
                    crawl_project_issue(project, jira)
                    processed_project.append(code)
                except (KeyboardInterrupt, Exception) as e:
                    traceback.print_exc()
                    print("Write processed project indices to files...")
                    utils.write_lines(processed_project, indices_file_path)
                    stop = True
                    break


crawl_jira_issue()
