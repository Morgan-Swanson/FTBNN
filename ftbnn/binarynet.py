import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 16

#static seed
seed = tf.random.normal([num_examples_to_generate, noise_dim])

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

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
        epochs=EPOCHS,
        validation_data=(test_images, test_labels),
        shuffle=True
    )

def train_gan(data, epochs, discriminator, generator):

    for epoch in range(epochs):
        start = time.time()

        train_step(data, discriminator, generator)

        predictions = generator(seed, training=False)

        show_image(predictions)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    show_image(predictions)

@tf.function
def train_step(images, discriminator, generator):
    noise = tf.random.normal([50000, noise_dim])

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


def show_image(predictions):
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    display.clear_output(wait=True)
    plt.show()



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # model.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 32, 32, 16)
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    print(model.output_shape)
    assert model.output_shape == (None, 32, 32, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

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
        './ftbnn_model.tf',
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None,
        options=None,
    )

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(tf.ones_like(fake_output), fake_output)

if __name__ == "__main__":
    #required to run on some systems
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    #init random noise    
    noise = tf.random.normal([1, 100])

    #init models
    generator = make_generator_model()
    discriminator = build_model()

    (train_images, train_labels), (test_images, test_labels) = preprocess_data()
    
    train_gan(train_images, EPOCHS, discriminator, generator)


    # generated_image = generator(noise, training=False)
    # decision = discriminator(generated_image)
    # print (decision)

    



    # model = build_model()
    # save_model(model)
    # trained_model = train_model(data, model)
    # plot_results(trained_model)