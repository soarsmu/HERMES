import utils
from jira import JIRA
import re
import urllib3
import data_loader
import traceback
from jira import JIRA, JIRAError
from entities import EntityEncoder, JiraTicket, JiraTicketIssueType, \
    JiraTicketComment, JiraTicketStatus, JiraTicketFixVersion, JiraTicketPriority, JiraTicketResolution

import json

jira_ticket_file_path = '../data/jira_ticket/'
entity_encoder = EntityEncoder()

urllib3.disable_warnings()
red_hat_username = 'yourusername'
red_hat_password = 'yourpassword'
jenkins_username = 'yourusername'
jenkins_password = 'yourpassword'

wrong_tagged_ticket_set = {'CVE-2018', 'CVE-2016', 'CVE-2012', 'CVE-2015', 'CVE-2013', 'CVE-2011',
                           'CVE-2009', 'CVE-2008', 'CVE-2010', 'CVE-2007', 'CVE-2014', 'UTF-8'}

not_exist_ticket_set = {'OCMPRESS-98', 'BUG-38897', 'MQ6-112', 'HUDSON-6437', 'APLO-366', 'BZ-1169553', 'HUDSON-8079',
                        'JSR-303', 'ISO-8859', 'HUDSON-2886', 'ONOS-4424', 'HUDSON-4820', 'JDK-8071638', 'GH-910',
                        'RFC-2616', 'GH-918', 'GH-637', 'GH-172', 'KIP-632', 'CESU-8', 'JOB1-212', 'HUDSON-1867',
                        'GH-1113', 'GH-865', 'MASTER-1059', 'HUDSON-6824', 'GG-9002', 'GH-1729', 'WFK-2', 'HUDSON-2379'}

no_permission_ticket_set = {'SOLR-8075', 'SOLR-12530', 'SECURITY-1186', 'SECURITY-726', 'SECURITY-1016',
                            'SECURITY-1129', 'SECURITY-643', 'SECURITY-1009', 'SECURITY-445', 'SECURITY-630',
                            'SECURITY-705', 'SECURITY-113', 'SECURITY-715', 'SECURITY-403', 'SECURITY-58',
                            'SECURITY-190', 'SECURITY-659', 'SECURITY-656', 'SECURITY-503', 'SECURITY-774',
                            'SECURITY-655', 'SECURITY-372', 'SECURITY-676', 'SECURITY-402', 'SECURITY-266',
                            'SECURITY-521', 'SECURITY-724', 'SECURITY-433', 'SECURITY-704', 'SECURITY-218',
                            'SECURITY-660', 'SECURITY-1071', 'SECURITY-498', 'SECURITY-79', 'SECURITY-499',
                            'SECURITY-260', 'SECURITY-75', 'SECURITY-73', 'SECURITY-904', 'SECURITY-717',
                            'SECURITY-93', 'SECURITY-55', 'SECURITY-466', 'SECURITY-49', 'SECURITY-108',
                            'SECURITY-1193', 'SECURIT-49', 'SECURITY-617', 'SECURITY-74', 'SECURITY-109',
                            'SECURITY-637', 'SECURITY-89', 'SECURITY-77', 'SECURITY-506', 'SECURITY-790',
                            'SECURITY-478', 'SECURITY-1072', 'SECURITY-595', 'SECURITY-667', 'SECURITY-996',
                            'SECURITY-611', 'SECURITY-514', 'SECURITY-1074', 'SPR-11426', 'KEYCLOAK-8260',
                            'KEYCLOAK-3667', 'JBPAPP-11251', 'UNDERTOW-438', 'JETTY-1042', 'OS-12',
                            'OS-13', 'BPMSPL-132', 'HUDSON-2324', 'SECURITY-429', 'JETTY-1084', 'SECURITY-532',
                            'JETTY-1133', 'WFK2-375'}


def write_jira_ticket_to_json(jira_tickets, record_id):
    file_path = jira_ticket_file_path + record_id + ".txt"
    json_dict_list = []
    for jira_ticket in jira_tickets:
        json_dict_list.append(entity_encoder.encode(jira_ticket))
    # this is for deserialization
    # jira_ticket_new = []
    # for json_dict in json_dict_list:
    #     jira_ticket_new.append(JiraTicket(json_value=json_dict))
    with open(file_path, 'w') as text_file:
        text_file.write(json.dumps(json_dict_list))


def extract_jira_tickets(records, jira):
    jira_ticket_regex = '((?!([A-Z0-9a-z]{1,10})-?$)[A-Z]{1}[A-Z0-9]+-\d+)'
    repo_to_ticket = {}
    repo_jira_set = set()
    record_to_ticket = {}
    for record in records:
        record_to_ticket[record] = set()
        repo = record.repo
        matched_list = re.findall(jira_ticket_regex, record.commit_message)
        for item in matched_list:
            if repo not in repo_to_ticket:
                repo_to_ticket[repo] = set()
            repo_to_ticket[repo].add(item[0])
            repo_jira_set.add(repo)
            record_to_ticket[record].add(item[0])
    return repo_jira_set, repo_to_ticket, record_to_ticket


def connect_to_jira(jira_server):
    print("jira server: {}".format(jira_server))
    jira = None
    if jira_server == "https://issues.redhat.com":
        jira = JIRA(server=jira_server, basic_auth=(red_hat_username, red_hat_password))
    elif jira_server == "https://issues.jenkins.io":
        # jira = JIRA(server=jira_server, basic_auth={jenkins_username, jenkins_password})
        jira = JIRA(server=jira_server)
    else:
        jira = JIRA(jira_server)
    return jira


def extract_jira_issue(jira, ticket, record):
    issue = jira.issue(ticket)
    fields = issue.fields

    summary = fields.summary
    description = fields.description
    created_at = fields.created
    creator = fields.creator.displayName

    assignee = None
    if fields.assignee is not None:
        assignee = fields.assignee.displayName

    fix_versions = []
    if hasattr(fields, 'fixVersions'):
        for fix_version in fields.fixVersions:
            if hasattr(fix_version, 'releaseDate'):
                fix_versions.append(JiraTicketFixVersion(name=fix_version.name, release_date=fix_version.releaseDate))
            else:
                fix_versions.append(JiraTicketFixVersion(name=fix_version.name))

    issue_type = JiraTicketIssueType(name=fields.issuetype.name, description=fields.issuetype.description)
    priority = JiraTicketPriority(priority_id=fields.priority.id, priority_name=fields.priority.name)
    resolution = None
    if fields.resolution is not None:
        resolution = JiraTicketResolution(resolution_id=fields.resolution.id,
                                          name=fields.resolution.name, description=fields.resolution.description)
    resolution_date = fields.resolutiondate
    status = JiraTicketStatus(name=fields.status.name, description=fields.status.description,
                              category=fields.status.statusCategory.name)
    comments = []
    for ticket_comment in fields.comment.comments:
        created_by = ticket_comment.author.displayName
        body = ticket_comment.body
        created_at = ticket_comment.created
        updated_at = ticket_comment.updated
        comments.append(JiraTicketComment(created_by=created_by, body=body,
                                          created_at=created_at, updated_at=updated_at))

    jira_ticket = JiraTicket(name=ticket, summary=summary, description=description,
                             created_at=created_at, creator=creator,
                             assignee=assignee, fix_versions=fix_versions, issue_type=issue_type, priority=priority,
                             resolution=resolution, resolution_date=resolution_date, status=status, comments=comments)
    return jira_ticket


def show_repo_not_in_dict(records, repo_to_jira_dictionary):
    repo_set = set()
    for record in records:
        repo_set.add(record.repo)

    repo_keys = repo_to_jira_dictionary.keys()
    for repo in repo_set:
        if repo not in repo_keys:
            print(repo)

    # with open('processed_jira_repos.txt', 'w') as f:
    #     for repo in repo_set:
    #         f.write(repo + '\n')


def process_jira_data():
    print('Processing jira...')
    file_name = '../MSR2019/experiment/full_dataset_with_patches.txt'
    records = data_loader.load_records(file_name)
    repo_jira_set, repo_to_ticket, record_to_ticket = extract_jira_tickets(records, None)

    repo_to_jira_file_name = '../repo_to_jira.txt'
    repo_to_jira_dictionary = {}

    # read repo_to_jira_set
    with open(repo_to_jira_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split('\t\t')
            repo_to_jira_dictionary[items[0]] = items[1][:-1]

    # show_repo_not_in_dict(records, repo_to_jira_dictionary)

    # read jira_to_repo_set
    jira_to_repo_dictionary = {}
    for key, value in repo_to_jira_dictionary.items():
        if value not in jira_to_repo_dictionary:
            jira_to_repo_dictionary[value] = set()
        jira_to_repo_dictionary[value].add(key)

    # create dictionary of jira connector
    jira_sever_dict = {}
    for key, value in jira_to_repo_dictionary.items():
        if key not in jira_sever_dict:
            jira = connect_to_jira(key)
            jira_sever_dict[key] = jira

    count = 0
    for record in records:
        count += 1
        print(record)
        repo = record.repo
        if repo in repo_to_jira_dictionary:
            jira = jira_sever_dict[repo_to_jira_dictionary[repo]]
            tickets = []
            for ticket in record_to_ticket[record]:
                if (ticket not in wrong_tagged_ticket_set
                        and ticket not in not_exist_ticket_set
                        and ticket not in no_permission_ticket_set):
                    try:
                        tickets.append(extract_jira_issue(jira, ticket, record))
                    except JIRAError:
                        print("Error...")
            if len(tickets) > 0:
                write_jira_ticket_to_json(tickets, record.id)

    print(len(records))
    print(count)
    # do some analysis

    # valid_ticket_set = set()
    # for jira_server, repo_set in jira_to_repo_dictionary.items():
    #     if jira_server not in {}:
    #         jira = connect_to_jira(jira_server)
    #         for repo in repo_set:
    #             for ticket in repo_to_ticket[repo]:
    #                 if (ticket not in wrong_tagged_ticket_set
    #                         and ticket not in not_exist_ticket_set
    #                         and ticket not in no_permission_ticket_set):
    #                     print(ticket)
    #                     extract_jira_issue(jira, ticket)
    #                     valid_ticket_set.add(ticket)
    #                     print("Connected to issue: {}".format(ticket))
    #         print_line_seperator()
    #
    # print_line_seperator()
    # print("Valid ticket count: {}".format(len(valid_ticket_set)))
    # enhanced_record_count = 0
    # enhanced_record = set()
    # for record, record_ticket_set in record_to_ticket.items():
    #     if len(valid_ticket_set.intersection(record_ticket_set)) > 0:
    #         enhanced_record_count += 1
    #         enhanced_record.add(record)
    # print("enhanced record count: {}".format(enhanced_record_count))
    # print_line_seperator()
    # return enhanced_record


process_jira_data()
