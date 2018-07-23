# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

"""CNN model define and train config params
"""

L2_REG_COEF = 1e-5
LEARNING_RATE = 1e-4
KERNEL_SIZES = [1,2,3]
NUM_OF_CNN_FILTERS = 128
DROPOUT_RATE = 0.4
DENSE_LAYER_SIZE = 128
MAX_SEQ_LENTH = 20
BATCH_SIZE = 256
N_EPOCHS = 20
RANDOM_STATE = 42