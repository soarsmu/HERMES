# HERMES
Auto Vulnernability Fix Commit Classification

Please checkout branch **master** for replication package

The dataset is available at location: MSR/2019/experiment/full_data_set_with_all_features.txt

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
