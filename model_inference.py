import torch
import os
import semi_deep_experiment


vector_length_limit = 100

HIDDEN_SIZE = 64
OUTPUT_SIZE = 2
EMBEDDING_DIMENSION = 64

directory = os.path.dirname(os.path.abspath(__file__))
model_folder_path = os.path.join(directory, 'model')
model_name = 'deep_model_2021-05-19_21_27_52.743265_fold_begin1.sav'
model = semi_deep_experiment.LSTMModel(EMBEDDING_DIMENSION, HIDDEN_SIZE, VOCABULARY_SIZE, OUTPUT_SIZE, BATCH_SIZE)

print(model.state_dict())