import os
import h5py
import torch
from utils import tensor_and_augment


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, split, transform=lambda x: x):
        if not os.path.exists(filename):
            raise IOError('Dataset "' + filename + '" does not exist')

        self.h5f = h5py.File(filename, 'r', libver='latest')
        self.data = self.h5f[split]
        self.transform = transform
        self.start = 0
        # note: use a valid dataset name to get the length of the dataset
        self.end = len(self.data['target'])

    def __len__(self):
        return self.end - self.start

    def _process_index(self, index, default):
        if index is None:
            index = default
        elif index < 0:
            index += self.end
        else:
            index += self.start

        return index

    def _process_slice(self, start=None, stop=None, step=None):
        return slice(self._process_index(start, self.start),
                     self._process_index(stop, self.end),
                     step)

    def __getitem__(self, slice):
        if isinstance(slice, int):
            slice = self._process_index(slice, self.start)
        else:
            slice = self._process_slice(slice.start, slice.stop, slice.step)

        data = {str(k): v[slice] for k, v in self.data.items()}

        return self.transform(data)

    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()

    def batches(self, batch_size, shuffle=False, size_limit=None):
        length = len(self)
        indices = torch.arange(0, length, batch_size)

        if size_limit is not None:
            indices = indices[:size_limit]

        if shuffle:
            indices = indices[torch.randperm(indices.size(0))]

        for start in indices:
            end = min(start + batch_size, length)
            yield self[start:end]


class dataset(HDF5Dataset):

    def __init__(self, filename, split, transform=None):

        self.split = split

        # if split not in ['training', 'validation', 'testing']:
        #     raise ValueError(
        #         self.split + ' is not a valid split, use one of training, validation or testing')

        super(dataset, self).__init__(filename, split, transform)


def get_dataset(filename, split='training', augment=False, device='cpu', synth=False, crop=False, resize=None):
    """Returns a HDF5Dataset of dict instances.

    Args:
            filename: a filename of an HDF5 file.
            split: one of "training", "validation" or "testing".
            augment: if set applies data augmentations.
            device: device on which to perform computations (usually "cuda" or "cpu").

    Returns:
            the HDF5Dataset of dict instances.
    """
    def transform(data):

        return tensor_and_augment(data, device=device, augment=augment,
                                  indices=['image_on', 'image_off'], synth=synth, crop=crop, resize=resize)

    return dataset(filename, split, transform)


if __name__ == '__main__':
    split = 'train'
    filename = 'dataset.h5'
    dataset = get_dataset('../data/processed/' + filename, split=split)
    print('%s dataset size: %d' % (split, len(dataset)))
