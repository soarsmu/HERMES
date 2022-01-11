import os
from data_loader import data_loader
import issue_linker
from entities import EntityEncoder
import utils
from utils import print_line_seperator


def write_dataset_with_enhanced_issue(sim_scores_file_name, enhanced_data_file_name, score_threshold, limit, drop_empty):
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/full_dataset_with_all_features.txt")
    record_with_enhanced_issue_file_path \
        = os.path.join(directory, enhanced_data_file_name)

    similarity_scores_file_path = os.path.join(directory, sim_scores_file_name)

    records = data_loader.load_records(record_file_path)

    jira_tickets = issue_linker.load_jira_tickets(testing=False)

    id_to_record = {}
    for record in records:
        id_to_record[int(record.id)] = record

    scores = []
    for line in utils.read_lines(similarity_scores_file_path):
        parts = line.split("\t\t")
        record_id = int(parts[0])
        if parts[2] == 'None':
            continue
        ticket_id = int(parts[2])
        score = float(parts[3])
        # ticket_key = parts[4]
        if score > 0:
            scores.append((record_id, ticket_id, score))

    print("Sorting scores...")
    scores.sort(key=lambda x: x[2], reverse=True)

    print("Finish sorting scores")

    count = 0
    for record_id, ticket_id, score in scores[:int((score_threshold * len(scores)))]:
        record = id_to_record[record_id]
        try:
            ticket = jira_tickets[ticket_id]
            record.jira_ticket_list.append(ticket)
        except IndexError:
            print(ticket_id)
        count += 1
        if limit != -1 and count == limit:
            break
        record = id_to_record[record_id]
        jira_ticket = jira_tickets[ticket_id]
        # print(score)
        # print(record.repo + '/commit/' + record.commit_id)
        # print("Ticket name: {}".format(jira_ticket.name))
        # print(jira_ticket.summary)
        # print(jira_ticket.description)
        # print("-------------------")
    # for record in records:
    #     if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
    #         count += 1
    #
    # print(count)
    if drop_empty:
        new_records = []
        for record in records:
            if len(record.jira_ticket_list) > 0 or len(record.github_issue_list) > 0:
                new_records.append(record)

        records = new_records
    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(records)

    print("Writing records...")
    with open(record_with_enhanced_issue_file_path, 'w') as file:
        file.write(json_value)
    print("Finishing writing")


def get_apache_subdataset(sim_scores_file_name, enhanced_data_file_name, score_threshold, limit):
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/full_dataset_with_all_features.txt")
    record_with_enhanced_issue_file_path \
        = os.path.join(directory, enhanced_data_file_name)

    similarity_scores_file_path = os.path.join(directory, sim_scores_file_name)

    records = data_loader.load_records(record_file_path)

    apache_repo_set = set()

    for line in utils.read_lines("repo_to_apache_key.txt"):
        parts = line.split("\t\t")
        apache_repo_set.add(parts[0])

    new_records = []
    for record in records:
        if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0 \
                or record.repo in apache_repo_set:
            if record.repo in apache_repo_set:
                print(2)
            new_records.append(record)

    records = new_records

    record_id_set = set()
    for record in records:
        record_id_set.add(int(record.id))

    jira_tickets = issue_linker.load_jira_tickets(testing=False)

    id_to_record = {}
    for record in records:
        id_to_record[int(record.id)] = record

    scores = []
    for line in utils.read_lines(similarity_scores_file_path):
        parts = line.split("\t\t")
        record_id = int(parts[0])
        if parts[2] == 'None':
            continue
        ticket_id = int(parts[2])
        score = float(parts[3])
        # ticket_key = parts[4]
        if record_id in record_id_set:
            print(record_id)
            scores.append((record_id, ticket_id, score))

    print("Sorting scores...")
    scores.sort(key=lambda x: x[2], reverse=True)

    print("Finish sorting scores")

    count = 0
    for record_id, ticket_id, score in scores[:int((score_threshold * len(scores)))]:
        record = id_to_record[record_id]
        try:
            ticket = jira_tickets[ticket_id]
            record.jira_ticket_list.append(ticket)
        except IndexError:
            print(ticket_id)
        count += 1
        if limit != -1 and count == limit:
            break
        record = id_to_record[record_id]
        jira_ticket = jira_tickets[ticket_id]
        # print(score)
        # print(record.repo + '/commit/' + record.commit_id)
        # print("Ticket name: {}".format(jira_ticket.name))
        # print(jira_ticket.summary)
        # print(jira_ticket.description)
        # print("-------------------")
    # for record in records:
    #     if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
    #         count += 1
    #
    # print(count)

    new_records = []
    for record in records:
        if len(record.github_issue_list) > 0 or len(record.jira_ticket_list) > 0:
            new_records.append(record)

    records = new_records

    entity_encoder = EntityEncoder()
    json_value = entity_encoder.encode(records)

    print("Writing records...")
    with open(record_with_enhanced_issue_file_path, 'w') as file:
        file.write(json_value)
    print("Finishing writing")


# get_apache_subdataset(sim_scores_file_name='sim_score_21042021.txt',
#                                   enhanced_data_file_name='MSR2019/experiment/enhanced_dataset_apache_repo_21042021_th_100.txt',
#                                   score_threshold=1,
#                                   limit=-1)

def do_something():
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/sub_enhanced_dataset_th_100.txt")

    records = data_loader.load_records(record_file_path)

    apache_repo_set = set()

    for line in utils.read_lines("repo_to_apache_key.txt"):
        parts = line.split("\t\t")
        apache_repo_set.add(parts[0])

    count = 0
    print(len(records))

    # new_records = []
    # count = 0
    # for record in records:
    #    if record.label == 1:
    #        count += 1
    #
    # print(count)
    # print(len(records))


def calculate_accuracy(threshold):
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/full_dataset_with_all_features.txt")

    records = data_loader.load_records(record_file_path)
    id_to_record = {}
    id_to_jira_label = {}
    scores = []
    for record in records:
        if len(record.jira_ticket_list) > 0:
            ticket = record.jira_ticket_list[0]
            id_to_jira_label[record.id] = ticket.name

    count_none = 0
    for line in utils.read_lines("texts/score_true_link_test.txt"):
        parts = line.split("\t\t")
        record_id = parts[0]
        predicted = parts[4]
        score = parts[3]
        scores.append((record_id, predicted, score))
        if predicted == "None":
            count_none+=1
    scores.sort(key=lambda x: x[2], reverse=True)
    print(len(scores))
    list_length = int(threshold * len(scores))
    print("List length: {}".format(list_length))
    correct = 0
    for record_id, predicted, score in scores[: list_length]:
        expected = id_to_jira_label[record_id]
        if predicted == id_to_jira_label[record_id]:
            correct += 1
    print("Correct: {}".format(correct))

    print("Accuracy: {}".format(correct/list_length))
    print("Count None: {}".format(count_none))

# 0.6 => 1002 => 1350 => 0.74
# 0.7 => 1091 => 1575 => 0.69
# 0.8 => 1151 => 1800 => 0.64
# 0.9 => 1206 => 2025 => 0.6
# 1   => 1229 => 2250 => 0.55

# do_something()

#
# write_dataset_with_enhanced_issue(sim_scores_file_name='sim_scores_relevant_repos.txt',
#                                   enhanced_data_file_name='MSR2019/experiment/enhanced_dataset_relevant_repos_01052021.txt',
#                                   score_threshold=1,
#                                   limit=-1)


def do_something_2():
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/enhanced_dataset_08042021_without_comments_th_100.txt")

    records = data_loader.load_records(record_file_path)
    id_to_record = {}
    for record in records:
        id_to_record[int(record.id)] = record

    issue_false = set()
    for line in utils.read_lines(os.path.join(directory, "MSR2019/experiment/statistics/false_positive/issue_2021-04-28_20_54_27.871173.txt")):
        issue_false.add(int(line))

    for line in utils.read_lines(os.path.join(directory, "MSR2019/experiment/statistics/false_negative/issue_2021-04-28_20_54_27.871173.txt")):
        issue_false.add(int(line))

    joint_false = set()
    for line in utils.read_lines(os.path.join(directory,
                                              "MSR2019/experiment/statistics/false_positive/joint_2021-04-28_20_54_27.871173.txt")):
        joint_false.add(int(line))

    for line in utils.read_lines(os.path.join(directory,
                                              "MSR2019/experiment/statistics/false_negative/joint_2021-04-28_20_54_27.871173.txt")):
        joint_false.add(int(line))

    false_set = issue_false.union(joint_false)
    true_set = set()
    for record in records:
        if int(record.id) not in false_set:
            true_set.add(int(record.id))

    count_pos = 0
    count_neg = 0
    old_records = data_loader.load_records(os.path.join(directory, "MSR2019/experiment/full_dataset_with_all_features.txt"))
    for record in old_records:
        if int(record.id) not in false_set and len(record.github_issue_list) == 0\
                and len(record.jira_ticket_list) == 0:
            if record.label == 1:
                count_pos += 1
                linked_record = id_to_record[int(record.id)]
                print(linked_record.repo + "/commit/" + linked_record.commit_id)
                ticket = linked_record.jira_ticket_list[0]
                key = ticket.name
                print(key)
                print("----------------------")
            if record.label == 0:
                count_neg += 1

    print(count_pos)
    print(count_neg)
    # apache_repo_set = set()
    #
    # for line in utils.read_lines("repo_to_apache_key.txt"):
    #     parts = line.split("\t\t")
    #     apache_repo_set.add(parts[0])
    #
    # count = 0
    # for record in records:
    #    if len(record.jira_ticket_list) > 0 or len(record.github_issue_list) > 0:
    #        count += 1
    # print(count)
    # new_records = []
    # count = 0
    # for record in records:
    #    if record.label == 1:
    #        count += 1
    #
    # print(count)
    # print(len(records))



# do_something()
# calculate_accuracy(0.5)


def do_something_3():
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/enhanced_dataset_08042021_without_comments_th_100.txt")

    records = data_loader.load_records(record_file_path)
    id_to_record = {}
    for record in records:
        id_to_record[int(record.id)] = record

    issue_false = set()
    issue_false_negative = set()
    for line in utils.read_lines(os.path.join(directory, "MSR2019/experiment/statistics/false_positive/issue_2021-04-28_20_54_27.871173.txt")):
        issue_false.add(int(line))

    for line in utils.read_lines(os.path.join(directory, "MSR2019/experiment/statistics/false_negative/issue_2021-04-28_20_54_27.871173.txt")):
        issue_false.add(int(line))
        issue_false_negative.add(int(line))

    joint_false = set()
    joint_false_negative = set()
    for line in utils.read_lines(os.path.join(directory,
                                              "MSR2019/experiment/statistics/false_positive/joint_2021-04-28_20_54_27.871173.txt")):
        joint_false.add(int(line))

    for line in utils.read_lines(os.path.join(directory,
                                              "MSR2019/experiment/statistics/false_negative/joint_2021-04-28_20_54_27.871173.txt")):
        joint_false.add(int(line))
        joint_false_negative.add(int(line))

    false_set = issue_false.union(joint_false)
    true_set = set()
    for record in records:
        if int(record.id) not in false_set:
            true_set.add(int(record.id))

    false_negative_set = issue_false_negative.union(joint_false_negative)

    count_pos = 0
    count_neg = 0
    old_records = data_loader.load_records(os.path.join(directory, "MSR2019/experiment/full_dataset_with_all_features.txt"))
    for record in old_records:
        if record.label == 1 and int(record.id) in false_negative_set and len(record.github_issue_list) == 0\
                and len(record.jira_ticket_list) == 0:
            count_pos += 1
            linked_record = id_to_record[int(record.id)]
            print(linked_record.repo + "/commit/" + linked_record.commit_id)
            ticket = linked_record.jira_ticket_list[0]
            key = ticket.name
            print(ticket.summary)
            print(ticket.description)
            print(key)
            print("----------------------")



# do_something_3()

# write_dataset_with_enhanced_issue(sim_scores_file_name='sim_scores_limit_feature_07042021.txt',
#                                   enhanced_data_file_name='MSR2019/experiment/sub_enhanced_dataset_th_80.txt',
#                                   score_threshold=0.8,
#                                   limit=-1,
#                                   drop_empty=True)



def analyze_dataset():
    directory = os.path.dirname(os.path.abspath(__file__))
    record_file_path = os.path.join(directory, "MSR2019/experiment/full_dataset_with_all_features.txt")
    records = data_loader.load_records(record_file_path)

    count = 0
    count_neg = 0
    miss = 0

    for record in records:
        if record.label == 1:
            print(record.repo + "/commit/" + record.commit_id)
            count += 1
            patch_length = 0
            for file in record.commit.files:
                patch = file.patch
                if patch is None:
                    miss += 1
                else:
                    patch_length += len(patch.split(" "))
            print(patch_length)
            print_line_seperator()

    print(miss)


# analyze_dataset()

write_dataset_with_enhanced_issue(sim_scores_file_name='sim_scores_limit_feature_07042021.txt',
                                  enhanced_data_file_name='MSR2019/experiment/sub_enhanced_dataset_th_90.txt',
                                  score_threshold=0.9,
                                  limit=-1,
                                  drop_empty=True)