import csv
import github
import traceback
import entities
from entities import Record

def clear_github_prefix(name):
    github_prefix = "https://github.com/"
    if name.startswith(github_prefix):
        name = name[len(github_prefix):]
    return name


def extract_record(file_name, has_message=False):
    records = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if has_message:
                record = Record(row[0], row[1], row[2], row[3], row[4])
            else:
                record = Record(row[0], row[1], row[2], "", row[3])
            records.append(record)
    return records


def read_lines(filepath):
    with open(filepath, "r") as f:
        content = f.readlines()
        return [x.strip() for x in content]


def write_lines(ids, filepath):
    with open(filepath, "w") as f:
        for id in ids:
            f.write(str(id) + '\n')
    f.close()


def print_line_seperator():
    print("------------------------------------------------")


def is_not_empty_commit(commit):
    return len(commit.files) > 0


def is_not_large_commit(commit):
    number_of_changes = 0
    number_of_addition = 0
    number_of_deletion = 0
    for file in commit.files:
        number_of_changes += file.changes
        number_of_addition += file.additions
        number_of_deletion += file.deletions
    # return number_of_changes <= 500 and number_of_addition <= 500 and number_of_deletion <= 500
    return number_of_changes <= 200 and number_of_addition <= 200 and number_of_deletion <= 200


def contain_java_file(commit):
    contain = False
    for file in commit.files:
        if file.filename.endswith('.java'):
            contain = True
    return contain


def process_github(repo_to_records):
    gh = github.Github("7dbf97a28908b738d31cc5b78a3fa6b7fb28929e")

    records_with_message = []
    processed_ids = read_lines('index.txt')
    for repo_name, records in repo_to_records.items():
        print(repo_name)
        try:
            repo = gh.get_repo(clear_github_prefix(repo_name))
            for record in records:
                index_id = record.repo + '/commit/' + record.commit_id
                if index_id in processed_ids:
                    continue
                commit_id = record.commit_id
                print("record id: {}".format(record.id))
                print(record.repo + "/commit/" + commit_id)
                try:
                    commit = repo.get_commit(commit_id)
                    commit_message = commit.commit.message
                    record.commit_message = commit_message
                    records_with_message.append(record)
                    processed_ids.append(index_id)
                except:
                    print("Something wrong with repo: {}, commit: {}".format(repo_name,commit_id))
                    write_lines(processed_ids, 'index.txt')
                    traceback.print_exc()
                    break
        except:
            print("Something wrong with repo: {}".format(repo_name))
            write_lines(processed_ids, 'index.txt')
            traceback.print_exc()
            break
        print_line_seperator()

    for record in records_with_message:
        record.id = int(record.id)
    records_with_message.sort(key=lambda x:x.id)

    return records_with_message


def write_to_csv(records, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for record in records:
            writer.writerow([record.id, record.repo, record.commit_id, record.commit_message, record.label])
