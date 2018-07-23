# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

"""

"""

import os
import re
from collections import defaultdict
from itertools import chain
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit


def get_train_pairs_from_description(data_description_path: str) -> List[Tuple[str, str]]:
    """

    :param data_description_path:
    :return:
    """
    result = []
    with open(data_description_path, 'r') as f:
        for line in f:
            result.append((line.split()[0], line.split()[-1]))
    return result


def get_labels_to_int_map(labels: str) -> Dict[str, int]:
    """

    :param labels:
    :return:
    """
    return dict(zip(sorted(list(set(labels))), range(len(labels))))


def get_test_pairs(test_data_path: str) -> List[Tuple[str, str]]:
    """

    :param test_data_path
    :return:
    """
    test_samples_pathes = os.listdir(test_data_path)
    test_samples_labels = [x.split('_')[0] for x in test_samples_pathes]
    return [(x, y) for x, y in zip(test_samples_pathes, test_samples_labels)]


def data_iterator(
        features: List[np.ndarray],
        labels: np.ndarray,
        batch_size: int,
        padder: object
) -> Tuple[np.ndarray, np.ndarray]:
    """batch iterator with data shuffle

    :param features: list of vecrtors sequences
    :param labels: array of shape (data_size, num_of_classes)
    :param batch_size: size of batch
    :param padder: special data preparer
    :return: Tuple(padded features batch, labels batch)
    """

    data_size = len(features)
    data_idx = np.array(list(range(data_size)))
    features = np.array(features)
    labels = labels
    while True:
        np.random.shuffle(data_idx)
        for i_start in np.arange(0, data_size, batch_size):
            batch_idx = data_idx[i_start: i_start + batch_size]
            yield padder.prepare(features[batch_idx]), labels[batch_idx]


class ToCNNdataTransformer(object):
    """

    """

    def __init__(self, max_seq_len: int) -> None:
        """Constructor

        :param max_sample_len: max lenth of vectors sequence
        """

        self._max_seq_len = max_seq_len

    def prepare(self, features: List[np.ndarray]) -> np.ndarray:
        """

        :param features: list of vectors sequences
        :return: padded features
        """

        vec_size = len(features[0][0])
        num_of_samples = len(features)
        box_shape = (num_of_samples, self._max_seq_len, vec_size)
        features_box = np.zeros(box_shape)

        for ind, sample_features in enumerate(features):
            truncate_features = sample_features[:self._max_seq_len]
            seq_len = len(truncate_features)
            features_box[ind, :seq_len] = truncate_features

        return features_box


def get_best_model_name(names: List[str]) -> str:
    """

    :param names: list of model names in specific format
    :return: model name with best accuracy
    """

    return sorted(names, key=lambda x: float(x.split('_')[4][:-5]))[-1]


def train_test_split(
        samples_paths: List[str],
        labels: List[int],
        seed: int = 42,
        test_size: float = 0.05
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Stratified shuffle split data to train / test, ignore augmentation

    :param samples_paths: list of samples paths List[str]
    :param labels: list of samples labels List[int]
    :param seed: random seed
    :param test_size: test fraq float
    :return: tuple(train_samples, train_labels, test_samples, test_labels)
    """

    def unroll_pair(pair: Tuple[int, List[str]]) -> List[Tuple[str, int]]:
        """Unroll pair

        :param pair: tuple(label, list of paths)
        :return: list of tuples (label, path)
        """

        label = pair[0]
        return list(zip(pair[1], [label] * len(pair[1])))

    pattern = re.compile('\d{4}')
    samples_groups_dict = defaultdict(list)

    for path, label in zip(samples_paths, labels):
        idx = pattern.search(path).group()
        samples_groups_dict[f'{idx}_{label}'].append(path)

    samples_groups = [(pair[0].split('_')[1], pair[1]) for pair in samples_groups_dict.items()]

    groups_labels = [g[0] for g in samples_groups]

    train_idx, test_idx = next(
        iter(
            StratifiedShuffleSplit(
                groups_labels,
                n_iter=1,
                test_size=test_size,
                random_state=seed
            )
        )
    )

    train_samples, train_labels = list(zip(*(chain(*[unroll_pair(samples_groups[idx]) for idx in train_idx]))))
    test_samples, test_labels = list(zip(*chain(*[unroll_pair(samples_groups[idx]) for idx in test_idx])))

    return train_samples, train_labels, test_samples, test_labels