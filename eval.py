# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

import logging
import os

import numpy as np
from keras.models import load_model
from tqdm import tqdm

from src.utils.data_utils import get_best_model_name
from src.utils.data_utils import get_labels_to_int_map
from src.utils.data_utils import get_test_pairs
from src.models.vggish_model import VggishModel
from src.features.vggish_input import wavfile_to_examples


test_data_path = 'data/test'
model_base_dirr = 'models'
model_save_pattern = 'cnn_model'
result_file_path = 'result.txt'
vgg_model_checkpoint = 'models/vggish_model.ckpt'

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    LOGGER.info('prepare test pairs')
    samples_pathes, labels = list(
        zip(
            *get_test_pairs(test_data_path)
        )
    )

    LOGGER.info('prepare labels map')
    labels_map = get_labels_to_int_map(labels)

    int_labels = np.array([labels_map[label] for label in labels])

    LOGGER.info('load Vggish model')
    vgg_model = VggishModel(vgg_model_checkpoint)

    LOGGER.info('prepare test vectors from data')
    test_root = 'data/test'
    test_vectors = [
        vgg_model.predict(
            wavfile_to_examples(os.path.join(test_root, sample_path))
        ) for sample_path in tqdm(samples_pathes, desc='vectors and features extracting')
    ]

    models_names = [file_name for file_name in os.listdir(model_base_dirr) if model_save_pattern in file_name]
    best_model_name = get_best_model_name(models_names)

    LOGGER.info(f'load cnn model name {best_model_name}')
    model = load_model(os.path.join(model_base_dirr, best_model_name), compile=False)

    LOGGER.info('predict')
    predict = [model.predict(v[np.newaxis, :])[0] for v in test_vectors]

    open_task_predict = [(np.argmax(x), x[np.argmax(x)]) for x in predict]

    inverse_label_map = dict((value, key) for key, value in labels_map.items())

    LOGGER.info('save results')
    with open(result_file_path, 'w') as f:
        for sample_path, (label, probability) in zip(samples_pathes, open_task_predict):
            f.write(f'{sample_path} {probability} {inverse_label_map[label]} \n')
