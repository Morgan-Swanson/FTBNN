import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocess_data():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    num_classes = 10
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
    test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

    # Normalize pixel values to be between -1 and 1
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
    return ((train_images, train_labels), (test_images, test_labels))

def build_model():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    # All quantized layers except the first will use the same options
    kwargs = dict(input_quantizer="ste_sign",
                  kernel_quantizer="ste_sign",
                  kernel_constraint="weight_clip",
                  use_bias=False)
    return tf.keras.models.Sequential([
        # In the first layer we only quantize the weights and not the input
        lq.layers.QuantConv2D(128, 3,
                              kernel_quantizer="ste_sign",
                              kernel_constraint="weight_clip",
                              use_bias=False,
                              input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantConv2D(128, 3, padding="same", **kwargs),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantConv2D(256, 3, padding="same", **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantConv2D(256, 3, padding="same", **kwargs),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantConv2D(512, 3, padding="same", **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantConv2D(512, 3, padding="same", **kwargs),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        tf.keras.layers.Flatten(),

        lq.layers.QuantDense(1024, **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantDense(1024, **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

        lq.layers.QuantDense(10, **kwargs),
        tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
        tf.keras.layers.Activation("softmax")
    ])

def train_model(training_data, model, epochs=10, batch_size=50):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    images, labels = training_data
    train_images, validation_images, train_labels, validation_labels = \
        train_test_split(images,
                         labels,
                         test_size=0.2,
                         random_state=42,
                         shuffle=True,
                         stratify=labels)
    model.compile(
        tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_images, validation_labels),
        shuffle=True
    )

def test_model(testing_data, model):
    testing_images, testing_labels = testing_data
    test_loss, test_acc = model.evaluate(testing_images, testing_labels)
    return test_acc

def plot_results(trained_model):
    plt.plot(trained_model.history['accuracy'])
    plt.plot(trained_model.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    print(np.max(trained_model.history['accuracy']))
    print(np.max(trained_model.history['val_accuracy']))

def save_model(trained_model, path="./models/"):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    tf.keras.models.save_model(
        trained_model,
        path + "ftbnn_model.tf",
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None,
        options=None,
    )

def load_model(path="./models/"):
    return tf.keras.models.load_model(path + 'ftbnn_model.tf')

if __name__ == "__main__":
    #required to run on some systems
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    model = build_model()
    training_data, testing_data = preprocess_data()
    history = train_model(training_data, model)
    save_model(model)
    plot_results(trained_model)
    accuracy = test_model(testing_data, model)
    print("Test Accuracy:{:.2%}".format(accuracy))
