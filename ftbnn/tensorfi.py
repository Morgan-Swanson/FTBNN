import TensorFI as ti
import tensorflow as tf

def insturment_model(model_path):
    tf.logging.set_verbosity(tf.logging.FATAL)
    sess = tf.Session()
    model = tf.keras.models.load_model('saved_model/my_model')
    fi = ti.TensorFI(sess, name = "logistReg", logLevel = 30, disableInjections = True)
