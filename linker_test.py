import issue_linker
import data_loader.data_loader as loader
import os
from issue_linker import *
import random

directory = os.path.dirname(os.path.abspath(__file__))

records = loader.load_records(os.path.join(directory, 'MSR2019/experiment/full_dataset_with_all_features.txt'))

# for record in records:
#     if len(record.jira_ticket_list) > 0:
#         print(record.repo + '/commit/' + record.commit_id)
#         for ticket in record.jira_ticket_list:
#             print(ticket.name)
#         input()

commit_id_list = ['83b38773a3e066f92d986d5181184b030b4ad502', 'e3769ea1d2d088b169a78a1a1127848b2b2b8cbd',
                  'f4ef817d255ecb471b4a6c5aed66dfd87329f355', 'd853853469292cd54fd9662c3605030ab5a9566b',
                  '94c0a6cfdf443712ad2c82c36b4e2ee4e99c4cb0', 'cfd470def51a962593dd29e24ab741b4fe2b90f2',
                  '30853ccbf5bf9cfdf5f0278318c88d7f35de57c9', '2d105a206884b62ccdba61f2de3e2fe65fc43074',
                  '927c017d228e1f4e756ea3e08f28a209677277b0', 'b6e69a616c0fc257fefa3c0be4942a47a7c51797']
ticket_id_list = ['PDFBOX-1337', 'YARN-829', 'HTTPCLIENT-1056', 'CAMEL-9309', 'WW-3984', 'CXF-4167',
                  'OFBIZ-644', 'HADOOP-14732', 'JCR-1443', 'JCR-1104']

new_records = []
for record in records:
    if record.commit_id in commit_id_list:
        new_records.append(record)
records = new_records

jira_tickets = []
count = 0
for record in records:
    for ticket in record.jira_ticket_list:
        if ticket.name in ticket_id_list:
            ticket.id = count
            count += 1
            jira_tickets.append(ticket)
            # if ticket.id in [8]:
            #     ticket.summary = None
            #     ticket.description = None
            # if ticket.id in [3]:
            #     ticket.summary = None
random.shuffle(records)
random.shuffle(jira_tickets)

using_code_terms_only = False
limit_feature = True
min_df = 0

for record in records:
    record.code_terms = extract_commit_code_terms(record)
    if not using_code_terms_only:
        record.text_terms_parts = extract_commit_text_terms_parts(record)

    for terms in record.text_terms_parts:
        if len(terms) <= 10:
            need_print = True

    # if record.id in ['4856']:
    #     record.code_terms = ''

for record in records:
    print(record.repo + '/commit/' + record.commit_id)
    print('Code terms')
    print(record.code_terms)
    print('Text terms')
    for terms in record.text_terms_parts:
        print(terms)
    print('-------------------------')

print("Start extracting issue features...")
for issue in jira_tickets:
    print(issue.id)
    issue.code_terms = extract_issue_code_terms(issue)
    if not using_code_terms_only:
        issue.text_terms_parts = extract_issue_text_terms_parts(issue, limit_feature)
    print(issue.name)
    print(issue.code_terms)
    print("Text terms:")
    for terms in issue.text_terms_parts:
        print(terms)
    print_line_seperator()
print("Finish extracting issue features")

if min_df > 1:
    tfidf_vectorizer = TfidfVectorizer(min_df=int(min_df))
elif min_df < 1:
    tfidf_vectorizer = TfidfVectorizer(min_df=float(min_df))
else:
    tfidf_vectorizer = TfidfVectorizer()

calculate_similarity_scores(records, jira_tickets, tfidf_vectorizer, using_code_terms_only)