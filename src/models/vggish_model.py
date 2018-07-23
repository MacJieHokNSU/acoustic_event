# Copyrights
#   Author: Alexey Svischev
#   Created: 09/07/2018

"""

"""

from typing import List

import numpy as np
import tensorflow as tf

from src.utils import load_vggish_slim_checkpoint
from src.utils import vggish_params



class VggishModel(object):
    """

    """

    def __init__(self, checkpoint_path: str) -> None:
        """

        :param checkpoint_path:
        """
        self.__sess = tf.Session()
        load_vggish_slim_checkpoint(self.__sess, checkpoint_path)
        self.__features_tensor = self.__sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME
        )
        self.__embedding_tensor = self.__sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )

    def predict(self, examples_batch: List[np.ndarray]) -> np.ndarray:
        """

        :param examples_batch:
        :return:
        """
        embedding_batch = self.__sess.run(
            self.__embedding_tensor,
            feed_dict={self.__features_tensor: examples_batch}
        )
        return embedding_batch
