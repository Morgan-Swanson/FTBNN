import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt


loss_object = tf.keras.losses.CategoricalCrossentropy()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def cross_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train_with_adversarial_pattern(data, batch, epochs, model, epsilon):
    (train_images, train_labels) = data
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for idx, img in enumerate(train_images):
            total_loss = train_adversarial_step(train_images[idx], train_labels[idx], model, epsilon)

        print('Epoch {:.4f} took {:.6f} sec. Average generated image accuracy is {:.6f}'.format(epoch + 1, time.time()-start, total_loss / batch))

@tf.function
def train_adversarial_step(image, label, model, epsilon):
    with tf.GradientTape() as img_tape:
        img_tape.watch(image)
        t_img = tf.expand_dims(image, axis=0)
        decision = model(t_img, training=True)
        loss = cross_loss(label, decision)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = img_tape.gradient(loss, image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)

    perturbed_img = image + epsilon*signed_grad
    p_img = tf.expand_dims(perturbed_img, axis=0)
    #retrain net on newly generated image
    pimg_prediction = model(p_img, training=True)

    return signed_grad