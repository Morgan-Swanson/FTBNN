                     name = "logistReg",
                     logLevel = 30,
                     disableInjections = True)

    fi = ti.TensorFI(sess,
    model = tf.keras.models.load_model('saved_model/my_model')
    tf.compat.v1.logging.set_verbosity(tf.logging.FATAL)
def insturment_model(model_path):
    sess = tf.Session()

import tensorflow as tf
import TensorFI as ti