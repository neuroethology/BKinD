import numpy as np

# Computed feature mean from the task 1 train split.
# Only needed for normalizing features.
feature_mean = "data/train_mean.npy"
feature_stdev = "data/train_stdev.npy"

def transpose_last_axis(orig_pose_dictionary, sequence_key = 'keypoints'):
  for key in orig_pose_dictionary:
    X = orig_pose_dictionary[key][sequence_key]

    if len(X.shape) == 4:
      X = X.transpose((0,1,3,2)) #last axis is x, y coordinates
      orig_pose_dictionary[key][sequence_key] = X
    else:
      # Assume the first 28 dims are the keypoints.
      transposed_X = X[:, :28].reshape((-1, 2, 2, 7)).transpose((0,1,3,2)) #last axis is x, y coordinates
      orig_pose_dictionary[key][sequence_key] = np.concatenate([
            transposed_X.reshape(-1, 28), X[:, 28:]], axis = -1) 

  return orig_pose_dictionary

def normalize_data(orig_pose_dictionary, sequence_key = 'keypoints'):
  """ 
  All sequences have 
  * Channel 0 with scale of 1024 
  * Channel 1 with scale of 570
  """
  for key in orig_pose_dictionary:
    X = orig_pose_dictionary[key][sequence_key]
    if len(X.shape) == 3:
      X[..., 0] = X[..., 0]/570
      X[..., 1] = X[..., 1]/1024
      orig_pose_dictionary[key][sequence_key] = X

    else:
      # Assume the first 28 dims are the keypoints.
      reshaped_X = X[:, :28].reshape((-1, 2, 7, 2))
      reshaped_X[..., 0] = reshaped_X[..., 0]/570
      reshaped_X[..., 1] = reshaped_X[..., 1]/1024

      mean = np.load(feature_mean)
      stdev = np.load(feature_stdev)

      norm_features = (X[:, 28:] - mean)/stdev

      orig_pose_dictionary[key][sequence_key] = np.concatenate([
          reshaped_X.reshape(-1, 28), norm_features], axis = -1) 
  return orig_pose_dictionary
