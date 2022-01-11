import click


class ExperimentOption():
    def __init__(self):
        self.data_set_size = -1
        self.ignore_number = True
        self.use_github_issue = True
        self.use_jira_ticket = True
        self.use_comments = True
        self.use_bag_of_word = True
        self.positive_weights = [0.5]
        self.max_n_gram = 1
        self.min_document_frequency = 1
        self.use_linked_commits_only = False

        # if self.use_issue_classifier = False, issue's information is attached to commit message
        self.use_issue_classifier = True

        self.fold_to_run = 10
        self.use_stacking_ensemble = True
        self.tf_idf_threshold = -1
        self.use_patch_context_lines = False
        self.unlabeled_size = -1


def read_option_from_command_line(size, unlabeled_size,
                                  ignore_number, github_issue, jira_ticket, use_comments,
                                  positive_weights, max_n_gram,
                                  min_document_frequency, use_linked_commits_only,
                                  use_issue_classifier,
                                  fold_to_run,
                                  use_stacking_ensemble,
                                  tf_idf_threshold,
                                  use_patch_context_line):
    experiment_option = ExperimentOption()
    experiment_option.data_set_size = size
    experiment_option.unlabeled_size = unlabeled_size
    experiment_option.ignore_number = ignore_number
    experiment_option.use_github_issue = github_issue
    experiment_option.use_jira_ticket = jira_ticket
    experiment_option.use_comments = use_comments
    experiment_option.positive_weights = list(positive_weights)
    experiment_option.max_n_gram = max_n_gram
    experiment_option.min_document_frequency = min_document_frequency
    experiment_option.use_linked_commits_only = use_linked_commits_only
    experiment_option.use_issue_classifier = use_issue_classifier
    experiment_option.fold_to_run = fold_to_run
    experiment_option.use_stacking_ensemble = use_stacking_ensemble
    experiment_option.tf_idf_threshold = tf_idf_threshold
    experiment_option.use_patch_context_lines = use_patch_context_line

    click.echo("Running process with these options:")
    if experiment_option.data_set_size == -1:
        click.echo("    Data set size: Full data")
    else:
        click.echo("    Data set size: {}".format(experiment_option.data_set_size))
    click.echo("    Ignore number as token: {}".format(experiment_option.ignore_number))
    click.echo("    Use github issue: {}".format(experiment_option.use_github_issue))
    click.echo("    Use jira ticket: {}".format(experiment_option.use_jira_ticket))
    click.echo("    Use comments: {}".format(experiment_option.use_comments))
    click.echo("    Use bag of words: {}".format(experiment_option.use_bag_of_word))
    click.echo("    Positive weights: {}".format(experiment_option.positive_weights))
    click.echo("    Max N-gram: {}".format(experiment_option.max_n_gram))
    click.echo("    Min document frequency: {}".format(experiment_option.min_document_frequency))
    click.echo("    Use linked commit only: {}".format(experiment_option.use_linked_commits_only))
    click.echo("    Use issue classifier: {}".format(experiment_option.use_issue_classifier))
    click.echo("    Fold to run: {}".format(experiment_option.fold_to_run))
    click.echo("    Use stacking ensemble: {}".format(experiment_option.use_stacking_ensemble))
    click.echo("    Tf-idf threshold: {}".format(experiment_option.tf_idf_threshold))
    click.echo("    Use patch context lines: {}".format(experiment_option.use_patch_context_lines))
    if experiment_option.unlabeled_size != -1:
        click.echo("    Unlabeled size: {}".format(experiment_option.unlabeled_size))
    return experiment_option
