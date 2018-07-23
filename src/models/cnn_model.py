# Copyrights
#   Author: Alexey Svischev
#   Created: 12/07/2018

from typing import List

from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam


def define_cnn_model(
        v_size: int,
        filters_cnn: int,
        kernel_sizes: List[int],
        dropout_rate: float,
        dense_size: int,
        num_of_classes: int,
        l2_reg: float,
        learning_rate: float
    ) -> Model:
    """ define and compile shallow and wide CNN model

    :param v_size: input vector size
    :param filters_cnn: num of cnn filters for each filter size
    :param kernel_sizes: cnn filters kernel sizes, List[int]
    :param dropout_rate: float, dropout regularization
    :param dense_size: dense layer size
    :param num_of_classes: size of output vector
    :param l2_reg: l2 regularization coef
    :param learning_rate: learnig rate coef
    :return: keras model
    """

    inp = Input(shape=(None, v_size))

    outputs = []

    for i in range(len(kernel_sizes)):

        output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes[i],
                          activation=None,
                          kernel_regularizer=l2(l2_reg), use_bias=False,
                          padding='same')(inp)
        output_i = BatchNormalization()(output_i)
        output_i = Activation('relu')(output_i)
        output_i = GlobalMaxPooling1D()(output_i)
        outputs.append(output_i)

    output = concatenate(outputs, axis=1)
    output = Dropout(rate=dropout_rate)(output)

    output = Dense(dense_size, activation=None, kernel_regularizer=l2(l2_reg), use_bias=False)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = Dense(num_of_classes, activation=None, kernel_regularizer=l2(l2_reg), use_bias=False)(output)
    output = BatchNormalization()(output)
    act_output = Activation('softmax')(output)

    model = Model(inputs=inp, outputs=act_output)

    model.compile(optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.95, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy',
                  metrics=[categorical_accuracy])
    return model
