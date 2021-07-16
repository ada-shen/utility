
import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def getDataFiles(list_filename):
    return [ line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def get_label_index(label, label_num = 0):
    number = len(label)
    label_index = np.zeros((number),dtype='int')
    count = 0
    for i in range(number):
        if label[i] == label_num:
            label_index[count] = i
            count += 1
    label_index[count:number] = [-1]*(number - count)
    return label_index, count


class ModelNetH5Dataset(object):
    def __init__(self, list_filename, batch_size = 32, npoints = 1024, shuffle=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.h5_files = getDataFiles(self.list_filename)
        self.reset()

    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))
        if self.shuffle: np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        #rotated_data = provider.random_rotate_batchdata(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        return provider.shuffle_points(rotated_data)


    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        self.current_data,self.current_label = load_h5(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label, _ = shuffle_data(self.current_data,self.current_label)

    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if augment: data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch

    def appoint_batch(self, dataset_idx, batch_idx, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        self.current_file_idx = dataset_idx
        self._load_data_file(self._get_data_filename())

        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        if augment: data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch

    def appoint_class_batch(self, dataset_idx, class_idx, batch_idx, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        self.current_file_idx = dataset_idx
        self.batch_idx = batch_idx
        self._load_data_file(self._get_data_filename())
        label_index, count = get_label_index(self.current_label, class_idx)
        class_data = self.current_data[label_index]
        class_label = self.current_label[label_index]
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = class_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = class_label[start_idx:end_idx].copy()
        if augment: data_batch = self._augment_batch_data(data_batch)
        return data_batch, label_batch

if __name__=='__main__':
    d = ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/train_files.txt')
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
