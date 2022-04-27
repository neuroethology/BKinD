import json
import os
import numpy as np 
import argparse

'''
Script for converting CalMS21 .json files into .npy files, with discovered keypoints.
Based on script provided by CalMS21 dataset: https://data.caltech.edu/records/1991

The .npy files have the same dictionary layout, except the entries are
numpy arrays instead of lists.
The final dictionary 'keypoint' entries will have shape:
sequence_length x 2 x 2 x 7.
'''

def convert_to_array(dictionary, keypoint_dir, feature_dictionary = None):
    # Convert dictionary values (lists) to numpy arrays, until depth 3.
    # If feature dictionary is not None, also concatenate the dictionary values.
    converted = {}


    directory = keypoint_dir

    counter = 0
    # First key is the group name for the sequences
    for groupname in dictionary.keys():

        converted[groupname] = {}
        # Next key is the sequence id
        for sequence_id in dictionary[groupname].keys():

            converted[groupname][sequence_id] = {}


            counter = counter + 1
            # If not adding features, add keypoints, scores, and annotations & metadata (if available)
            if feature_dictionary is None:

                print(sequence_id)
                data = np.load(os.path.join(directory,
                        sequence_id.split('/')[-1] + '.seq.npz'))['keypoints']

                data_conf = np.load(os.path.join(directory,
                    sequence_id.split('/')[-1] + '.seq.npz'))['confidence']

                data_covs = np.load(os.path.join(directory,
                    sequence_id.split('/')[-1] + '.seq.npz'))['covs']                

                converted[groupname][sequence_id]['keypoints'] = np.concatenate([data, data_conf[:, :, np.newaxis], data_covs], axis = -1)
                print(data.shape)
            else:
                keypoints = np.array(dictionary[groupname][sequence_id]['keypoints'])
                converted[groupname][sequence_id]['features'] = np.concatenate([keypoints.reshape(keypoints.shape[0], -1),
                                                feature_dictionary[groupname][sequence_id]['features']], axis = -1)

            converted[groupname][sequence_id]['scores'] = np.array(dictionary[groupname][sequence_id]['scores'])         
               
            if 'annotations' in dictionary[groupname][sequence_id].keys():
                converted[groupname][sequence_id]['annotations'] = np.array(dictionary[groupname][sequence_id]['annotations'])                         

            if 'metadata' in dictionary[groupname][sequence_id].keys():
                converted[groupname][sequence_id]['metadata'] = dictionary[groupname][sequence_id]['metadata']                  

    print(counter)
    return converted


def json_save_to_npy(input_name, output_name, keypoint_dir, feature_name = None):
    with open(input_name, 'r') as fp:
        input_data = json.load(fp)

    input_data = convert_to_array(input_data, keypoint_dir)

    print("Saving " + output_name)
    np.save(output_name, input_data, allow_pickle=True)    



def convert_all_calms21(args):

    calms21_files = [args.input_train_file, args.input_test_file]
    input_dirs = [args.keypoint_dir_train, args.keypoint_dir_test]

    for i, single_file in enumerate(calms21_files):

        file_name = single_file.split('/')[-1].split('.')[0]
        npy_output_name = os.path.join(args.output_directory, file_name)
        json_save_to_npy(single_file, npy_output_name, input_dirs[i])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_file', type=str, required = False, 
    	help='Path to CalMS21 Task 1 train file')
    parser.add_argument('--input_test_file', type=str, required = False, 
        help='Path to CalMS21 Task 1 test file')    
    parser.add_argument('--keypoint_dir_train', type=str, required = False, 
        help='Directory to discovered keypoints for train split')
    parser.add_argument('--keypoint_dir_test', type=str, required = False, 
        help='Directory to discovered keypoints for test split')                
    parser.add_argument('--output_directory', type=str, default = 'data', required = False, 
    	help='Directory to output npy files')    

    parsed_args = parser.parse_args()

    convert_all_calms21(parsed_args)
