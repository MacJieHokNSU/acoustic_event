# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

"""

"""

import logging
import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from tqdm import tqdm

from src.utils import cnn_params
from src.utils.data_utils import get_labels_to_int_map
from src.utils.data_utils import get_train_pairs_from_description
from src.utils.data_utils import ToCNNdataTransformer
from src.utils.data_utils import train_test_split
from src.utils.data_utils import data_iterator
from src.models.cnn_model import define_cnn_model
from src.models.vggish_model import VggishModel
from src.features.vggish_input import wavfile_to_examples


LOGGER = logging.getLogger('train log')
logging.basicConfig(level=logging.INFO)

data_description_path = 'data/meta/meta.txt'
model_save_path = 'models/cnn_model_{epoch:02d}_{val_loss:.5f}_{val_categorical_accuracy:.5f}.hdf5'
vgg_model_checkpoint = 'models/vggish_model.ckpt'


if __name__ == '__main__':

    LOGGER.info('prepare data pairs')
    samples_pathes, labels = list(
        zip(
            *get_train_pairs_from_description(data_description_path)
        )
    )

    LOGGER.info('prepare labels map')
    labels_map = get_labels_to_int_map(labels)
    int_labels = [labels_map[label] for label in labels]

    LOGGER.info('split data to train test')
    train_samples, train_labels, test_samples, test_labels = train_test_split(
        samples_pathes,
        int_labels,
        seed=cnn_params.RANDOM_STATE
    )

    LOGGER.info('load Vggish model')
    vgg_model = VggishModel(vgg_model_checkpoint)

    LOGGER.info('prepare features from data')
    train_root = 'data/audio'

    train_features = [
        wavfile_to_examples(os.path.join(train_root, sample_path))
        for sample_path in tqdm(train_samples, desc='train features extracting')
    ]

    test_features = [
        wavfile_to_examples(os.path.join(train_root, sample_path))
        for sample_path in tqdm(test_samples, desc='test features extracting')
    ]

    LOGGER.info('prepare train vectors')
    train_vectors, train_labels = list(
        zip(
            *[(vgg_model.predict(x), y) for x, y in
                tqdm(zip(train_features, train_labels), desc='train vectors extracting') if len(x) > 0]
        )
    )

    LOGGER.info('prepare test vectors')
    test_vectors, test_labels = list(
        zip(
            *[(vgg_model.predict(x), y) for x, y in
              tqdm(zip(test_features, test_labels), desc='test vectors extracting') if len(x) > 0]
        )
    )

    vector_size = len(train_vectors[0][0])
    num_of_classes = len(np.unique(labels))

    LOGGER.info('prepare data to cnn input (padding)')
    data_padder = ToCNNdataTransformer(cnn_params.MAX_SEQ_LENTH)

    train_it = data_iterator(
        train_vectors,
        to_categorical(train_labels),
        cnn_params.BATCH_SIZE,
        data_padder)

    test_it = data_iterator(
        test_vectors,
        to_categorical(test_labels),
        cnn_params.BATCH_SIZE,
        data_padder)

    LOGGER.info('define CNN model')
    model = define_cnn_model(
        v_size=vector_size,
        filters_cnn=cnn_params.NUM_OF_CNN_FILTERS,
        kernel_sizes=cnn_params.KERNEL_SIZES,
        dropout_rate=cnn_params.DROPOUT_RATE,
        dense_size=cnn_params.DENSE_LAYER_SIZE,
        num_of_classes=num_of_classes,
        l2_reg=cnn_params.L2_REG_COEF,
        learning_rate=cnn_params.LEARNING_RATE
    )

    model_saver = ModelCheckpoint(
        model_save_path,
        monitor='val_acc',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1
    )

    LOGGER.info('train CNN model')
    model.fit_generator(
        train_it,
        steps_per_epoch=len(train_vectors) // cnn_params.BATCH_SIZE + 1,
        validation_data=test_it,
        validation_steps=len(test_vectors) // cnn_params.BATCH_SIZE + 1,
        epochs=cnn_params.N_EPOCHS,
        verbose=2,
        callbacks=[model_saver]
    )
