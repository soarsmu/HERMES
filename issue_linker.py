from loader import data_loader
import os
import json
from entities import JiraTicket
import re
from utils import print_line_seperator
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import data_preprocessor
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import click
import random

stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

directory = os.path.dirname(os.path.abspath(__file__))
jira_ticket_file_path = os.path.join(directory, 'data/jira_issue_batch_data')
similarity_scores_file_path = os.path.join(directory, 'texts/similarity_scores.txt')

source_code_extensions = ['.ios', '.c', '.java7', '.scala', '.cpp', '.php', '.cc', '.js', '.html',
                         '.swift', '.h', '.java', '.css']

c_notation_re = '[A-Za-z]+[0-9]*_.*'
qualified_name_re = '[A-Za-z]+[0-9]∗[\\.].+'
camel_case_re = '[A-Za-z]+.*[A-Z]+.*'
upper_case_re = '[A-Z0-9]+'
system_variable_re = '_+[A-Za-z0-9]+.+'
reference_expression_re = '[a-zA-Z]+[:]{2,}.+'

non_alphanumeric_pattern = re.compile(r'\W+', re.UNICODE)
hyper_link_pattern = re.compile(r"http\S+", re.UNICODE)
contain_both_number_and_char_pattern = re.compile(r'^(?=.*[a-zA-Z])(?=.*[0-9])', re.UNICODE)
regex = re.compile('|'.join([c_notation_re, qualified_name_re,
                             camel_case_re, upper_case_re,
                             system_variable_re, reference_expression_re]))

terms_min_length = 0

chunk_size = -1

repo_to_key = {}
apache_key_set = set()
lines = utils.read_lines('repo_to_apache_key.txt')
for line in lines:
    parts = line.split("\t\t")
    repo = parts[0]
    repo_to_key[repo] = []
    keys = parts[1].split(',')
    for key in keys:
        apache_key_set.add(key)
        repo_to_key[repo].append(key)
    repo_to_key[repo] = tuple(repo_to_key[repo])
apache_keys = tuple(apache_key_set)

use_relevant_ticket = False

# Write all file names to text file so reading order is deterministic
def write_all_ticket_file_names():
    file_names = []
    for file_name in os.listdir(jira_ticket_file_path):
        if file_name.endswith('.txt'):
            file_names.append(file_name)

    utils.write_lines(file_names, "jira_tickets_file_names.txt")


def load_jira_tickets(testing):
    print("Start loading crawled Jira tickets...")
    jira_tickets = []
    # todo count here is just for testing
    count = 0

    file_names = utils.read_lines(os.path.join(directory, 'jira_tickets_file_names.txt'))
    id_count = 0
    for file_name in file_names:
        if use_relevant_ticket and not file_name.startswith(apache_keys):
            continue
        # todo count here is just for testing
        count += 1
        if testing:
            if count == 10:
                break

        with open(jira_ticket_file_path + '/' + file_name) as file:
            json_raw = file.read()
            json_dict_list = json.loads(json_raw)
            for json_dict in json_dict_list:
                if json_dict is not None and json_dict != 'null':
                    id_count += 1
                    ticket = JiraTicket(json_value=json_dict)
                    ticket.id = id_count
                    jira_tickets.append(ticket)
    print("Finished loading crawled Jira ticket")
    return jira_tickets


# jira_ticket_list = load_jira_ticket()
all_file_extension = set()


def retrieve_code_terms(text):
    match_terms = []
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if not line.startswith("import") and not line.startswith("- import") and not line.startswith("+ import"):
            new_lines.append(line)

    text = " ".join(new_lines)
    tokens = word_tokenize(text)
    # tokens = text.split(" ")

    for token in tokens:
        if re.fullmatch(regex, token) and not token.isnumeric():
            # lowercase all token and split terms by '.' e.g, dog.speakNow -> ['dog', 'speakNow']
            parts = token.split('.')
            for part in parts:
                match_terms.extend(data_preprocessor.camel_case_split(part))

    match_terms = [token.lower() for token in data_preprocessor.under_score_case_split(match_terms)]
    match_terms = [token for token in match_terms if token not in stopwords_set]
    match_terms = [stemmer.stem(token) for token in match_terms]

    return match_terms


def extract_commit_code_terms(record):
    terms = []
    retrieve_code_terms(record.commit_message)
    for file in record.commit.files:
        if file.patch is None:
            continue
        terms.extend(retrieve_code_terms(file.patch))

    return " ".join(terms)


def extract_issue_code_terms(issue):
    terms = []

    if issue.description is not None:
        terms.extend(retrieve_code_terms(issue.description))

    if issue.summary is not None:
        terms.extend(retrieve_code_terms(issue.summary))

    for comment in issue.comments:
        terms.extend(retrieve_code_terms(comment.body))

    return " ".join(terms)


def extract_text(text):
    # filter hyperlinks
    # remove non-sense lengthy token, e.g 13f79535-47bb-0310-9956-ffa450edef68
    # remove numeric token
    # remove token contains both number(s) and character(s), e.g ffa450edef68
    raw_tokens = [token for token in text.split(' ') if not re.fullmatch(hyper_link_pattern, token)
                  # and not re.fullmatch(regex, token)
                  and not token.isnumeric()
                  and len(token) < 20
                  and not re.fullmatch(contain_both_number_and_char_pattern, token)]

    text = " ".join(raw_tokens)


    # todo check if non-alphanumeric characters removal is necessary
    text = non_alphanumeric_pattern.sub(' ', text)

    tokens = word_tokenize(text)

    code_terms = retrieve_code_terms(text)
    tokens.extend(code_terms)

    tokens = [token for token in tokens if not re.fullmatch(regex, token)
              and not token.isnumeric()
              and len(token) < 20
              and not re.fullmatch(contain_both_number_and_char_pattern, token)]


    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token not in stopwords_set]

    tokens = [stemmer.stem(token) for token in tokens]

    if len(tokens) < terms_min_length:
        return []

    if chunk_size == -1:
        return [" ".join(tokens)]

    parts = []
    index = 0
    while index < len(tokens):
        if len(tokens) - index < terms_min_length:
            break

        parts.append(" ".join(tokens[index:min(index + chunk_size, len(tokens))]))
        index += chunk_size

    return parts


def is_non_source_document(file_name):
    for extension in source_code_extensions:
        if file_name.endswith(extension):
            return False

    return True


def extract_commit_text_terms_parts(record):
    terms_parts = []

    text_term = extract_text(record.commit_message)
    if len(text_term) > 0:
        terms_parts = text_term

    for file in record.commit.files:
        if is_non_source_document(file.file_name) and file.patch is not None:
            text_term = extract_text(file.patch)
            if text_term is not None:
                terms_parts.extend(text_term)

    return terms_parts


def extract_issue_text_terms_parts(issue, limit_feature):
    terms_parts = []

    if issue.description is not None:
        text_term = extract_text(issue.description)
        if len(text_term) > 0:
            terms_parts.extend(text_term)

    if issue.summary is not None:
        text_term = extract_text(issue.summary)
        if len(text_term) > 0:
            terms_parts.extend(text_term)

    if not limit_feature:
        for comment in issue.comments:
            text_term = extract_text(comment.body)
            if len(text_term) > 0:
                terms_parts.extend(text_term)

    return terms_parts


def get_tfidf_for_words(tfidf_matrix, feature_names, corpus_index):
    # get tfidf values from matrix instead of transform text => save time
    feature_index = tfidf_matrix[corpus_index, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[corpus_index, x] for x in feature_index])
    score_dict = {}
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        score_dict[w] = s
    return score_dict


def calculate_similarity(record_term_scores, ticket_term_scores):
    term_set = set()

    if len(record_term_scores) == 0 or len(ticket_term_scores) == 0:
        return 0

    if len(set(record_term_scores.keys()) & set(ticket_term_scores.keys())) == 0:
        return 0

    for term, value in record_term_scores.items():
        term_set.add(term)
    for term, value in ticket_term_scores.items():
        term_set.add(term)

    term_to_record_score = {}
    term_to_ticket_score = {}

    for term in term_set:
        if term in record_term_scores:
            term_to_record_score[term] = record_term_scores[term]
        if term in ticket_term_scores:
            term_to_ticket_score[term] = ticket_term_scores[term]

    # calculate cosine similarity
    numerator = 0
    for term in term_set:
        if term in record_term_scores and term in ticket_term_scores:
            numerator += term_to_record_score[term]*term_to_ticket_score[term]

    sub1 = 0
    for term, value in term_to_record_score.items():
        sub1 += value ** 2
    sub1 = math.sqrt(sub1)

    sub2 = 0
    for term, value in term_to_ticket_score.items():
        sub2 += value ** 2
    sub2 = math.sqrt(sub2)

    denominator = sub1 * sub2
    score = numerator / denominator
    return score


def link_similarity(record, ticket, corpus_id_to_tf_idf_score,
                    record_id_to_corpus_id, ticket_id_to_corpus_id):
    record_corpus_ids = sorted(record_id_to_corpus_id[record.id])
    ticket_corpus_ids = sorted(ticket_id_to_corpus_id[ticket.id])

    # first documents are code terms documents
    max_score = 0

    if len(corpus_id_to_tf_idf_score[record_corpus_ids[0]]) >= terms_min_length \
            and len(corpus_id_to_tf_idf_score[ticket_corpus_ids[0]]) >= terms_min_length:
        max_score = calculate_similarity(corpus_id_to_tf_idf_score[record_corpus_ids[0]],
                                         corpus_id_to_tf_idf_score[ticket_corpus_ids[0]])
    # if max_score == 0:
    #     return 0

    # calculate text terms similarity scores
    for record_document_id in record_corpus_ids[1:]:
        if len(corpus_id_to_tf_idf_score[record_document_id]) < terms_min_length:
            continue

        for ticket_document_id in ticket_corpus_ids[1:]:
            if len(corpus_id_to_tf_idf_score[ticket_document_id]) < terms_min_length:
                continue

            max_score = max(max_score, calculate_similarity(corpus_id_to_tf_idf_score[record_document_id],
                                                            corpus_id_to_tf_idf_score[ticket_document_id]))
    return max_score


def calculate_corpus_document_score(tfidf_matrix, feature_names, corpus):
    id_to_score = {}
    for index in range(len(corpus)):
        id_to_score[index] = get_tfidf_for_words(tfidf_matrix, feature_names, index)

    return id_to_score


def calculate_similarity_scores(records, jira_tickets, tfidf_vectorizer, using_code_terms_only):
    corpus = []
    corpus_index = -1

    record_id_to_corpus_id = {}
    for record in records:
        corpus_index += 1
        record_id_to_corpus_id[record.id] = [corpus_index]
        corpus.append(record.code_terms)

        if not using_code_terms_only:
            for text_terms in record.text_terms_parts:
                corpus_index += 1
                record_id_to_corpus_id[record.id].append(corpus_index)
                corpus.append(text_terms)

    ticket_id_to_corpus_id = {}
    for ticket in jira_tickets:
        corpus_index += 1
        ticket_id_to_corpus_id[ticket.id] = [corpus_index]
        corpus.append(ticket.code_terms)

        if not using_code_terms_only:
            for text_terms in ticket.text_terms_parts:
                corpus_index += 1
                ticket_id_to_corpus_id[ticket.id].append(corpus_index)
                corpus.append(text_terms)

    print("Calculating TF-IDF vectorizer...")
    # tfidf_vectorizer.fit(corpus)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names()
    print("Finish calculating TF-IDF vectorizer")

    print("Start calculating TF-IDF score for every words in every document in corpus...")
    corpus_id_to_tfidf_score = calculate_corpus_document_score(tfidf_matrix, feature_names, corpus)
    print("Finish calculating TF-IDF score")

    score_lines = []
    record_count = 0
    for record in records:
        if use_relevant_ticket and record.repo not in repo_to_key:
            continue
        record_count += 1

        # if record.commit_id != '4dd6206547de8f694532579e37ba8103bafaeb1':
        #     continue

        max_score = 0
        best_ticket = None
        for ticket in jira_tickets:
            if use_relevant_ticket and not ticket.name.startswith(repo_to_key[record.repo]):
                continue
            current_score = link_similarity(record, ticket, corpus_id_to_tfidf_score,
                                            record_id_to_corpus_id, ticket_id_to_corpus_id)
            if current_score > max_score:
                max_score = current_score
                best_ticket = ticket
        if best_ticket is not None:
            score_lines.append(str(record.id) + '\t\t' + record.repo + '/commit/' + record.commit_id + '\t\t'
                               + str(best_ticket.id)
                               + '\t\t' + str(max_score) + '\t\t' + best_ticket.name)
        else:
            score_lines.append(
                str(record.id) + '\t\t' + record.repo + '/commit/' + record.commit_id + '\t\t' + 'None'
                + '\t\t' + '0' + '\t\t' + 'None')

        if record_count % 50 == 0:
            print("Finished {} records".format(record_count))

    utils.write_lines(score_lines, similarity_scores_file_path)


@click.command()
@click.option('--testing', default=False, type=bool)
@click.option('--min_df', default=1, type=int)
@click.option('--using_code_terms_only', default=False, type=bool)
@click.option('--limit-feature', default=False, type=bool)
@click.option('--text-feature-min-length', default=0, type=int)
@click.option('--output-file-name', default='texts.txt', type=str)
@click.option('--chunk', default=- 1, type=int)
@click.option('--relevant-ticket', default=True, type=bool)
@click.option('--test-true-link', default=False, type=bool)
@click.option('--merge-link', default=False, type=bool)
@click.option('--max_df', default=1, type=float)
def process_linking(testing, min_df, using_code_terms_only, limit_feature, text_feature_min_length, output_file_name,
                    chunk, relevant_ticket, test_true_link, merge_link, max_df):

    # test_true_link is option for testing how many percent of records in our dataset link to their real issues
    # merge_link is option to choose whether we merge "real issues" to "crawled issues" to check the ability of
    # issue linker to recover true link

    global terms_min_length
    terms_min_length = text_feature_min_length

    global similarity_scores_file_path
    similarity_scores_file_path = os.path.join(directory, output_file_name)

    global chunk_size
    chunk_size = chunk

    global use_relevant_ticket
    use_relevant_ticket = relevant_ticket

    print("Setting:")
    print("     Testing: {}".format(testing))
    print("     Min document frequency: {}".format(min_df))
    print("     Max document frequency: {}".format(max_df))
    print("     Using code terms only: {}".format(using_code_terms_only))
    print("     Limit feature: {}".format(limit_feature))
    print("     Text terms min length: {}".format(terms_min_length))
    print("     Output file name: {}".format(output_file_name))
    print("     Chunk size: {}".format(chunk_size))
    print("     Use relevant ticket: {}".format(use_relevant_ticket))
    print("     Test true link: {}".format(test_true_link))
    print("     Merge link: {}".format(merge_link))
    print_line_seperator()

    records = data_loader.load_records(os.path.join(directory, 'MSR2019/experiment/full_dataset_with_all_features.txt'))

    new_records = []
    for record in records:
        # if testing => get the first jira issue of record
        if test_true_link:
            if len(record.jira_ticket_list) > 0:
                new_records.append(record)
        else:
            if len(record.jira_ticket_list) == 0 and len(record.github_issue_list) == 0:
                if use_relevant_ticket and record.repo not in repo_to_key:
                    continue
                new_records.append(record)

    records = new_records
    # random.shuffle(records)

    print("Records length: {}".format(len(records)))
    # todo for testing only
    if testing:
        records = records[:1000]
        pass

    print("Start extract commit features...")
    short_term_count = 0
    for record in records:
        # if record.commit_id != 'fed39c3619825bd92990cf1aa7a4e85119e00a6e':
        #     continue
        record.code_terms = extract_commit_code_terms(record)
        # record.code_terms = ''
        if not using_code_terms_only:
            record.text_terms_parts = extract_commit_text_terms_parts(record)

        need_print = False
        for terms in record.text_terms_parts:
            if len(terms) <= 10:
                need_print = True
        # if record.commit == '4dd6206547de8f694532579e37ba8103bafaeb1':
        #     print(record.repo + '/commit/' + record.commit_id)
        #     print(record.code_terms)
        #     print("Text terms:")
        #     for terms in record.text_terms_parts:
        #         print(terms)
        #     print_line_seperator()
        #     input()

        if not need_print:
            continue
        short_term_count += 1

    print("Finish extract commit features")
    print(short_term_count)

    jira_tickets = []
    if test_true_link:
        # if merge with crawled corpus, ticket_id must be assign from "lasted" issues id + 1
        if merge_link:
            jira_tickets = load_jira_tickets(testing)
            jira_tickets = jira_tickets[:30000]
            current_count = len(jira_tickets)
            for record in records:
                ticket = record.jira_ticket_list[0]
                current_count += 1
                ticket.id = current_count
                jira_tickets.append(ticket)
        # else issue id count from 0
        else:
            current_count = 0
            for record in records:
                ticket = record.jira_ticket_list[0]
                current_count += 1
                ticket.id = current_count
                jira_tickets.append(ticket)
    else:
        jira_tickets = load_jira_tickets(testing)

    # random.shuffle(jira_tickets)
    print("Issues length: {}".format(len(jira_tickets)))
    print("Start extracting issue features...")
    for issue in jira_tickets:
        issue.code_terms = extract_issue_code_terms(issue)
        # issue.code_terms = ''
        if not using_code_terms_only:
            issue.text_terms_parts = extract_issue_text_terms_parts(issue, limit_feature)

        # if issue.name in ['CAMEL-16146', 'HADOOP-14246']:
        #     print(issue.name)
        #     print(issue.code_terms)
        #     print("Text terms:")
        #     for terms in issue.text_terms_parts:
        #         print(terms)
        #     print_line_seperator()
        #     input()
    print("Finish extracting issue features")
    tfidf_vectorizer = TfidfVectorizer()
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    if max_df != 1:
        tfidf_vectorizer.max_df = max_df

    calculate_similarity_scores(records, jira_tickets, tfidf_vectorizer, using_code_terms_only)


if __name__ == '__main__':
    process_linking()
