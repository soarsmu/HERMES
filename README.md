# HERMES
Auto Vulnernability Fix Commit Classification

The dataset is available at location: MSR/2019/experiment/full_data_set_with_all_features.txt

Watch our presentation at SANER 2022: https://www.youtube.com/watch?v=S4a3wpHbVTw

Slides for the presentation in file: HERMES-SANER-2022_slides.pptx

Also PDF version: HERMES-SANER-2022_slides.pdf

in json format

Data Object Classes are described in **entitites.py**

###### To replicate HERMES on full dataset please use this command
python3 experiment.py --min_df 5 --use_linked_commits_only False --use_issue_classifier True --use_stacking_ensemble True --use-patch-context-lines False --tf-idf-threshold 0.005 --dataset full_dataset_with_all_features.txt

###### To replicate HERMES on subset of explicitly linked commits please use this command
python3 experiment.py --min_df 5 --use_linked_commits_only True --use_issue_classifier True --use_stacking_ensemble True --use-patch-context-lines False --tf-idf-threshold 0.005 --dataset full_dataset_with_all_features.txt

##### To replicate HERMES on the explicitly linked data and on commits recovered links

Please extract zip files in MSR2019/experiment which contain dataset corresponding to different threshold (thresholds are written at postfix of file names).

After that, to run HERMES on different threshold, please use this command's template:

python3 experiment.py --min_df 5 --use_linked_commits_only True --use_issue_classifier True --use_stacking_ensemble True --use-patch-context-lines False --tf-idf-threshold 0.005 --dataset **file_name**

where file_name is name of file in list of files just extracted

Link to our issue corpus: https://zenodo.org/record/5602211#.YXjQg9ZBxO8

Descriptions for parameter in command:
- min_df [real]: Min document frequency to filter out infrequent terms
- use_linked_commits_only [boolean]: Option to use all commits in dataset for training and testing, or only use commits where each contain at least one issue linked to Github or Jira issue tracker
- use_issue_classifier [boolean]: Option to use or not use issue classifier in HERMES. If not, HERMES contain only message classifier and patch classifier
- use_stacking_ensemble [boolean]: Option to use stacking ensemble or simple voting to combine base classifier (i.e. message classsifier, patch classifier, issue classifier). If true, use stacking ensemble for combination. Otherwise, use simple voting
- tf-idf-threshold [real]: Option in issue classifier to filter out noises in issue classifier
- dataset [string]: Name of the dataset selected for experiment
