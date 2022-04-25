import numpy as np


def load_mabe_data_task1(data_path):
    """ 
    Load data for task 1:
        Task 1 has multiple sequences
        The vocaubulary denotes the behavior name to class number.
        The vocabulary is all the same for all sequences in this task.
    """
    data_dict = np.load(data_path, allow_pickle=True).item()
    dataset = data_dict['annotator-id_0']
    # Get any sequence key.
    sequence_id = list(data_dict['annotator-id_0'].keys())[0]
    vocabulary = data_dict['annotator-id_0'][sequence_id]['metadata']['vocab']
    return dataset, vocabulary


def load_mabe_data_task2(data_path):
    """ 
    Load data for task 2:
        Task 2 has multiple sequences for multiple annotators
        Each annotator has annotations on a different set of sequences
        The vocaubulary denotes the behavior name to class number.
        The vocabulary is all the same for all sequences in this task.        
    """
    dataset = np.load(data_path, allow_pickle=True).item()

    # Get any sequence key.
    sequence_id = list(dataset['annotator-id_1'].keys())[0]
    vocabulary = dataset['annotator-id_1'][sequence_id]['metadata']['vocab']

    return dataset, vocabulary


def load_mabe_data_task3(data_path):
    """ 
    Load data for task 3:
        Task 3 has multiple sequences to different behaviors.
        Each sequence is labelled with binary labels.
    """
    dataset = np.load(data_path, allow_pickle=True).item()
    return dataset
