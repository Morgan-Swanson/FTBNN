import tensorflow as tf
import numpy as np
import time

noise_dim = 100
num_examples_to_generate = 16

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

def train_gan(data, batch, epochs, discriminator, generator):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    (train_images, train_labels), (validation_images, validation_labels) = data

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        array_split = (len(train_images) / batch) 
        for data_batch in np.array_split(train_images, array_split):
            total_loss += train_gan_step(data_batch, batch, (discriminator, discriminator_optimizer), (generator, generator_optimizer))

        #verification?
        print('Epoch {:.4f} took {:.6f} sec. Average generated image accuracy is {:.6f}'.format(epoch + 1, time.time()-start, total_loss / batch))

        # prediction = generator(seed, training=False)
        # show_image(prediction)

    # Generate after the final epoch
    #prediction = generator(seed, training=False)
    #show_image(prediction)

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
    return disc_loss

def make_dcgan_model():
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
