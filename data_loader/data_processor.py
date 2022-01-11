from entities import Record, JiraTicket, GithubIssue, EntityEncoder
import json
from json import JSONEncoder

jira_ticket = JiraTicket('jira title', 'jira description', 'jira created at')
github_issue = GithubIssue('github title', 'github description', 'github created at')
record = Record('1', 'test_repo', 'test_commit', 'test_commit_message', 'pos')
record.add_jira_ticket(jira_ticket)
record.add_github_ticket(github_issue)

record_encoder = EntityEncoder()
json_value = record_encoder.encode(record)
new_record = Record(json_value=json_value)
print(new_record)
print(new_record.github_issue_list)
print(new_record.jira_ticket_list)
