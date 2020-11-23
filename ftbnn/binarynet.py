import tensorflow as tf
#this must be done first/BEFORE GPU VIRTUAL DEVICES ARE LOADED
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 2GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import ftbnn.dcgan as dcgan
import ftbnn.fgsmgan as fgsmgan

EPOCHS = 10
EPSILON = 0.01
BATCH_SIZE = 50
model_path = '../models/'

def preprocess_data():
   num_classes = 10

   (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

   train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
   test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

   # Normalize pixel values to be between -1 and 1
   train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

   #generate labels from dataset
   train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
   test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

   #stratify train dataset and reserve test set
   train_images, validation_images, train_labels, validation_labels = \
      train_test_split(train_images,
                        train_labels,
                        test_size=0.2,
                        random_state=42,
                        stratify=train_labels)

   return (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels)

def train_model(training_data, model, epochs=10):
    (train_images, train_labels), (validation_images, validation_labels) = training_data

    model.compile(
        tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model.fit(
        train_images,
        train_labels,
        batch_size=50,
        epochs=epochs,
        validation_data=(validation_images, validation_labels),
        shuffle=True
    )

def show_image(predictions):
    plt.imshow(predictions[0, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.show()

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), use_bias=False, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(128, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2D(256, (3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    # model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dense(10))
    tf.keras.layers.Activation("softmax")

    return model

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
        model_path + 'ftbnn_model.tf',
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None,
        options=None,
    )

if __name__ == "__main__":
    #init models
    print('Generating models...')
    gan_generator = dcgan.make_dcgan_model()

    ftbnn = build_model()

    print('Loading data...')
    (train_images, train_labels), \
    (validation_images, validation_labels), \
    (test_images, test_labels) = preprocess_data()

    print('Training model...')
    training_data = (train_images, train_labels)
    #dcgan.train_gan(training_data, BATCH_SIZE, EPOCHS, ftbnn, gan_generator)
    fgsmgan.train_with_adversarial_pattern(training_data, BATCH_SIZE, EPOCHS, ftbnn, EPSILON)

    # model = build_model()
    # training_data, testing_data = preprocess_data()
    # history = train_model(training_data, model)    
    # plot_results(trained_model)
    # save_model(model)
    # accuracy = test_model(testing_data, model)
    # print("Test Accuracy:{:.2%}".format(accuracy))
