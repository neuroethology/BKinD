from tensorflow import keras
import numpy as np


def calculate_input_dim(feature_dim, architechture, past_frames, future_frames):
    """
    Data is arranged as [t, flattened_feature_dimensions]
           where t => [past_frames + 1 + future_frames]

    In this version, we flatten the feature dimensions
    But another generator, inherited from this class,
    could very well retain the actual structure of the mice
    coordinates.
    """
    flat_dim = np.prod(feature_dim)
    if architechture != 'fully_connected':
        input_dim = ((past_frames + future_frames + 1), flat_dim,)
    else:
        input_dim = (flat_dim * (past_frames + future_frames + 1),)
    return input_dim


def mabe_generator(data, augment, shuffle, sequence_key, kwargs):
    if data is not None:
        return MABe_Data_Generator(data,
                               augment=augment,
                               shuffle=shuffle,
                               sequence_key=sequence_key,
                               **kwargs)
    else:
        return None


class MABe_Data_Generator(keras.utils.Sequence):
    """
    Generates window of frames from sequence data
    * Each window comprises of past and future frames
    * Frame skip > 1 can be used to increased for subsampling
    * Augments by rotation and shifting frames
    * Boundaries are padded with zeros for when window exceeds the limits
    """
    def __init__(self,  pose_dict,
                 class_to_number,
                 batch_size=2,
                 input_dimensions=(2, 2, 7),
                 augment=False,
                 past_frames=100,
                 future_frames=100,
                 frame_skip=1,
                 shuffle=True,
                 sequence_key = 'keypoints'):

        self.batch_size = batch_size
        self.dim = input_dimensions

        self.classname_to_index_map = class_to_number
        self.n_classes = len(self.classname_to_index_map)

        self.past_frames = past_frames
        self.future_frames = future_frames
        self.frame_skip = frame_skip

        self.shuffle = shuffle
        self.augment = augment

        self.sequence_key = sequence_key

        # Raw Data Containers
        self.X = {}
        self.y = []

        # Setup Dimensions of data points
        # self.setup_dimensions()

        # Load raw pose dictionary
        self.load_pose_dictionary(pose_dict)

        # Setup Utilities
        self.setup_utils()

        # Generate a global index of all datapoints
        self.generate_global_index()

        # Epoch End preparations
        self.on_epoch_end()

    def load_pose_dictionary(self, pose_dict):
        """ Load raw pose dictionary """
        self.pose_dict = pose_dict
        self.video_keys = list(pose_dict.keys())

    def setup_utils(self):
        """ Set up padding utilities """
        self.setup_padding_utils()

    def setup_padding_utils(self):
        """ Prepare to pad frames """
        self.left_pad = self.past_frames * self.frame_skip
        self.right_pad = self.future_frames * self.frame_skip

        if self.sequence_key == 'keypoints':
            self.pad_width = (self.left_pad, self.right_pad), (0, 0), (0, 0)
        else:
            self.pad_width = (self.left_pad, self.right_pad), (0, 0)

    def classname_to_index(self, annotations_list):
        """
        Converts a list of string classnames into numeric indices
        """
        return np.vectorize(self.classname_to_index_map.get)(annotations_list)

    def generate_global_index(self):
        """ Define arrays to map video keys to frames """
        self.video_indices = []
        self.frame_indices = []

        self.action_annotations = []

        # For all video keys....
        for video_index, video_key in enumerate(self.video_keys):
            # Extract all annotations
            annotations = self.pose_dict[video_key]['annotations']
            # add annotations to action_annotations
            self.action_annotations.extend(annotations)

            number_of_frames = len(annotations)

            # Keep a record for video and frame indices
            # Keep a record of video_indices
            self.video_indices.extend([video_index] * number_of_frames)
            # Keep a record of frame indices
            self.frame_indices.extend(range(number_of_frames))
            # Add padded keypoints for each video key
            self.X[video_key] = np.pad(
                self.pose_dict[video_key][self.sequence_key], self.pad_width)

        self.y = np.array(self.action_annotations)
        # self.y = self.classname_to_index(self.action_annotations) # convert text labels to indices
        self.X_dtype = self.X[video_key].dtype  # Store D_types of X

        # generate a global index list for all data points
        self.indices = np.arange(len(self.frame_indices))

    def __len__(self):
        ct = len(self.indices) // self.batch_size
        ct += int((len(self.indices) % self.batch_size) > 0)
        return ct

    def get_X(self, data_index):
        """
        Obtains the X value from a particular global index
        """
        # Obtain video key for this datapoint
        video_key = self.video_keys[
            self.video_indices[data_index]
        ]
        # Identify the (local) frame_index
        # to offset original data padding
        frame_index = self.frame_indices[data_index] + self.left_pad
        # Slice from beginning of past frames to end of future frames
        slice_start_index = frame_index - self.left_pad
        slice_end_index = frame_index + self.frame_skip + self.right_pad
        assert slice_start_index >= 0
        _X = self.X[video_key][
            slice_start_index:slice_end_index:self.frame_skip
        ]
        if self.augment:
            _X = self.augment_fn(_X)
        return _X

    def augment_fn(self, to_augment):
        """ 
        Augment sequences
            * Rotation - All frames in the sequence are rotated by the same angle
                using the euler rotation matrix
            * Shift - All frames in the sequence are shifted randomly
                but by the same amount
        """
        if len(to_augment.shape) != 4:
            x = to_augment[:, :28].reshape(-1, 2, 7, 2)
        else:
            x = to_augment

        # Rotate
        angle = (np.random.rand()-0.5) * (np.pi * 2)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        x = np.dot(x, rot)

        # Shift - All get shifted together
        shift = (np.random.rand(2)-0.5) * 2 * 0.25
        x = x + shift

        if len(to_augment.shape) != 4:

            x = np.concatenate([x.reshape(-1, 28), to_augment[:, 28:]], axis = -1)

        return x

    def __getitem__(self, index):
        batch_size = self.batch_size
        batch_indices = self.indices[
            index*batch_size:(index+1)*batch_size]  # List indexing overflow gets clipped

        batch_size = len(batch_indices)  # For the case when list indexing is clipped

        X = np.empty((batch_size, *self.dim), self.X_dtype)

        for batch_index, data_index in enumerate(batch_indices):
            # Obtain the post-processed X value at the said data index
            _X = self.get_X(data_index)
            # Reshape the _X to the expected dimensions
            X[batch_index] = np.reshape(_X, self.dim)

        y_vals = self.y[batch_indices]
        # Converting to one hot because F1 callback needs one hot
        y = keras.utils.to_categorical(y_vals, num_classes=self.n_classes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":

    pose_dict_path = "../datasets/mabe_task1_data.npy"
    pose_dict = np.load(pose_dict_path, allow_pickle=True).item()

    generator = MABe_Data_Generator(
        pose_dict,
        batch_size=2,
        feature_dimensions=(2, 2, 7),
        classes=['other', 'investigation', 'attack', 'mount'],
        past_frames=100,
        future_frames=100,
        frame_skip=1,
        shuffle=True)

    print(generator[0])
    print("Length : ", len(generator))
    for X, y in generator:
        pass
