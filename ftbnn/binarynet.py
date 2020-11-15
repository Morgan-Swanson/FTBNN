import tensorflow as tf
#this must be done first/BEFORE GPU VIRTUAL DEVICES ARE LOADED
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


import larq as lq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

EPOCHS = 10
BATCH_SIZE = 50
noise_dim = 100
num_examples_to_generate = 16
model_path = '../models/'

#static seed
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


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

def train_gan(data, epochs, discriminator, generator):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    (train_images, train_labels), (validation_images, validation_labels) = data

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        array_split = (len(train_images) / BATCH_SIZE) 
        for data_batch in np.array_split(train_images, array_split):
            total_loss += train_gan_step(data_batch, BATCH_SIZE, (discriminator, discriminator_optimizer), (generator, generator_optimizer))

        #verification?
        print('Epoch {} took {} sec. Average generated image accuracy is {}'.format(epoch + 1, time.time()-start, total_loss / BATCH_SIZE))

        # prediction = generator(seed, training=False)
        # show_image(prediction)

    # Generate after the final epoch
    prediction = generator(seed, training=False)
    show_image(prediction)

@tf.function
def train_gan_step(images, batch, discriminator_data, generator_data):
    (discriminator, discriminator_optimizer) = discriminator_data
    (generator, generator_optimizer) = generator_data

    noise = tf.random.normal([batch, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss


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

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8*8*48, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((8, 8, 48)))
    assert model.output_shape == (None, 8, 8, 48) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(48, (3,3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 48)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(12, (3,3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 12)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (3,3), strides=(2, 2), padding='same', use_bias=False, activation='softmax')) #tanh
    #print(model.output_shape)
    assert model.output_shape == (None, 32, 32, 3) #3072 6/512

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
    generator = make_generator_model()
    discriminator = build_model()

    print('Loading data...')
    (train_images, train_labels), \
    (validation_images, validation_labels), \
    (test_images, test_labels) = preprocess_data()

    training_data = (train_images, train_labels), (validation_images, validation_labels)
    train_gan(training_data, EPOCHS, discriminator, generator)


    # model = build_model()
    # training_data, testing_data = preprocess_data()
    # history = train_model(training_data, model)    
    # plot_results(trained_model)
    # save_model(model)
    # accuracy = test_model(testing_data, model)
    # print("Test Accuracy:{:.2%}".format(accuracy))
