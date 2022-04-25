from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def num_to_text(number_to_class, anno_list):
    """ 
    Convert list of class numbers to list of class names
    """
    return np.vectorize(number_to_class.get)(anno_list)


def split_data(orig_pose_dictionary, vocabulary, seed=2021,
               test_size=0.5, split_videos=False):
    """ 
    Split data into train and test:
    * Full sequences are either put into train or test to avoid data leakage
    * By default, the "attack" behavior's presence is used to stratify the split
    * Optionally, the sequences may be split into half and treated as separate sequences
    """

    if test_size == 0.0:
        return orig_pose_dictionary, None

    number_to_class = {v: k for k, v in vocabulary.items()}
    if split_videos:
        pose_dictionary = {}
        for key in orig_pose_dictionary:
            key_pt1 = key + '_part1'
            key_pt2 = key + '_part2'
            anno_len = len(orig_pose_dictionary[key]['annotations'])
            split_idx = anno_len//2
            pose_dictionary[key_pt1] = {
                'annotations': orig_pose_dictionary[key]['annotations'][:split_idx],
                'keypoints': orig_pose_dictionary[key]['keypoints'][:split_idx]}
            pose_dictionary[key_pt2] = {
                'annotations': orig_pose_dictionary[key]['annotations'][split_idx:],
                'keypoints': orig_pose_dictionary[key]['keypoints'][split_idx:]}
    else:
        pose_dictionary = orig_pose_dictionary

    def get_percentage(sequence_key):
        anno_seq = num_to_text(
            number_to_class, pose_dictionary[sequence_key]['annotations'])
        counts = {k: np.mean(np.array(anno_seq) == k) for k in vocabulary}
        return counts

    anno_percentages = {k: get_percentage(k) for k in pose_dictionary}

    anno_perc_df = pd.DataFrame(anno_percentages).T

    rng_state = np.random.RandomState(seed)
    try:
        idx_train, idx_val = train_test_split(anno_perc_df.index,
                                              stratify=anno_perc_df['attack'] > 0,
                                              test_size=test_size,
                                              random_state=rng_state)
    except:
        idx_train, idx_val = train_test_split(anno_perc_df.index,
                                              test_size=test_size,
                                              random_state=rng_state)

    train_data = {k: pose_dictionary[k] for k in idx_train}
    val_data = {k: pose_dictionary[k] for k in idx_val}
    return train_data, val_data
