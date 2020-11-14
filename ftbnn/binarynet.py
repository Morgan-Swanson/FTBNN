import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data():
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

def train_model(data, model):
    (train_images, train_labels), (test_images, test_labels) = data
    model.compile(
        tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model.fit(
        train_images,
        train_labels,
        batch_size=50,
        epochs=10,
        validation_data=(test_images, test_labels),
        shuffle=True
    )

def plot_results(trained_model):
    plt.plot(trained_model.history['accuracy'])
    plt.plot(trained_model.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    print(np.max(trained_model.history['accuracy']))
    print(np.max(trained_model.history['val_accuracy']))

def save_model(trained_model):
    tf.keras.models.save_model(
        trained_model,
        './ftbnn_model.tf',
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None,
        options=None,
    )

if __name__ == "__main__":
    #required to run on some systems
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    model = build_model()
    save_model(model)
    trained_model = train_model(preprocess_data(), model)
    plot_results(trained_model)