import glob
import re

import torch
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import numpy as np
import pandas as pd


class LocationsDataset(Dataset):
    """Dataset housing pairs of game state images and agent locations.

    Attributes:
        images    = [ndarray] 4D array of environment state images
        locations = [ndarray] 2D array of agent locations
    """

    def __init__(self, images, locations):
        """
        Args:
            images    = [ndarray] 4D array of environment state images
            locations = [ndarray] 2D array of agent locations
        """
        self.images = images
        self.locations = locations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.locations[index]


def expand_data(states, coordinates, transformer):
    """Increase the number of elements in the data set.

    Attributes:
        states      = [ndarray] 4D array of environment state images
        coordinates = [ndarray] 2D array of locations of the agents
        transformer = [Transformer] object that transforms images

    Returns [(ndarray, ndarray)]:
        The expanded set of images and locations, respectively. The images
        include the original image and 8 extra images made from that original.
        They are flipped left to right, top to bottom, over the diagonal,
        and translated five times. The locations are changed accordingly.
    """
    images = []
    locations = []

    for img, positions in zip(states, coordinates):
        # add original image
        images.append(img)
        locations.append(positions)

        # add transposed image
        img_transposed = transformer.transpose(img)
        positions_transposed = (positions[1::-1], positions[3:1:-1])
        positions_transposed = np.concatenate(positions_transposed)
        images.append(img_transposed)
        locations.append(positions_transposed)

        # add flipped images
        img_flipped = transformer.flip(img, 'left_right')
        positions_flipped = positions.copy()
        positions_flipped[:4:2] = (transformer.size + 1) - positions[:4:2]
        images.append(img_flipped)
        locations.append(positions_flipped)

        img_flipped = transformer.flip(img, 'top_bottom')
        positions_flipped = positions.copy()
        positions_flipped[1:4:2] = (transformer.size + 1) - positions[1:4:2]
        images.append(img_flipped)
        locations.append(positions_flipped)

        # add translated images
        num_translations = 5
        while num_translations > 0:
            axis = np.random.choice(2)
            shift = np.random.choice([-6, -5, -4, -3, 3, 4, 5, 6])

            if axis == 0:  # vertical axis
                positions_translated = positions.copy()
                positions_translated[1:4:2] += shift
            else:  # horizontal axis
                positions_translated = positions.copy()
                positions_translated[:4:2] += shift

            if max(positions_translated) <= transformer.size and \
               min(positions_translated) >= 1:  # still within stage boundaries
                img_translated = transformer.translate(img, axis, shift)
                images.append(img_translated)
                locations.append(positions_translated)
                num_translations -= 1

    print(f'Number of training examples: {len(images)}')

    return np.array(images), np.array(locations)


def split_data(images, locations, train_split, test_split):
    """Split the data into training, validation, and testing data sets.

    Args:
        images      = [ndarray] 4D array of environment state images
        locations   = [ndarray] 2D array of locations of the agents
        train_split = [float] percentage of data that is for training
        test_split  = [float] percentage of data that is for testing

    Returns:
        The data splits as three tuples. The validation
        split is determined as `1 - train_split - test_split`.
    """
    # determine permutation to control labeler bias
    permutation = np.random.permutation(len(images))

    # determine maximum percentiles of training and validation splits
    train_split = int(train_split * len(images))
    validation_split = int((1 - test_split) * len(images))

    # select the images for each split
    train_images = images[permutation[:train_split]]
    validation_images = images[permutation[train_split:validation_split]]
    test_images = images[permutation[validation_split:]]

    # select the locations for each split
    train_locations = locations[permutation[:train_split]]
    validation_locations = locations[permutation[train_split:validation_split]]
    test_locations = locations[permutation[validation_split:]]

    return ((train_images, train_locations),
            (validation_images, validation_locations),
            (test_images, test_locations))


def natural_key(img_file_name):
    """Function that retrieves the integer in the image file name. This
       is used to sort the images to align them with the row numbers
       in `../data/locations_overlaps.csv`.

    Args:
        img_file_name = [str] the file name of an image

    Returns [int]:
        Integer in image file name.
    """
    return int(re.search(r'\d+', img_file_name).group())


def load_data(transformer, batch_size=40, train_split=0.64, test_split=0.2,
              data_dir='../data/', device=torch.device('cpu')):
    """Load the image and locations data from storage.

    Args:
        transformer = [Transformer] object that transforms images
        batch_size  = [int] number of data points in a batch
        train_split = [float] percentage of data that is for training
        test_split  = [float] percentage of data that is for testing
        data_dir    = [str] file path for directory with images and locations
        device      = [torch.device] device to put the data on

    Returns [(DataLoader,)]:
        The training, validation, and testing dataloaders.
    """
    # get the images
    image_file_names = glob.iglob(f'{data_dir}images/*.png')
    image_file_names = sorted(image_file_names, key=natural_key)
    images = [Image.open(f).convert('RGB') for f in image_file_names]
    images = np.array([np.asarray(i, 'f').transpose((2, 0, 1)) for i in images])

    # get the locations
    locations_overlaps = pd.read_csv(f'{data_dir}locations_overlaps.csv')
    locations_overlaps = locations_overlaps.to_numpy(dtype=np.float32)
    locations = locations_overlaps[:, :4]

    # split data set into appropriate subsets
    data = split_data(images, locations, train_split, test_split)
    train_data, validation_data, test_data = data

    # increase number of elements in the training data set
    train_data = expand_data(train_data[0], train_data[1], transformer)

    # put images and locations on GPU
    train_images = torch.tensor(train_data[0], device=device)
    train_locations = torch.tensor(train_data[1], device=device)
    validation_images = torch.tensor(validation_data[0], device=device)
    validation_locations = torch.tensor(validation_data[1], device=device)
    test_images = torch.tensor(test_data[0], device=device)
    test_locations = torch.tensor(test_data[1], device=device)

    # make data sets for each split
    train_set = LocationsDataset(train_images, train_locations)
    validation_set = LocationsDataset(validation_images, validation_locations)
    test_set = LocationsDataset(test_images, test_locations)

    # make batch iterators for each split
    train_iter = DataLoader(train_set, batch_size, shuffle=True)
    validation_iter = DataLoader(validation_set, batch_size, shuffle=False)
    test_iter = DataLoader(test_set, batch_size, shuffle=False)

    return train_iter, validation_iter, test_iter
