"""
Proposed face reconstruction GAN

Author: Pleško Filip <iplesko@fit.vut.cz>
Date: May 14, 2023
"""
from abc import ABC
from tensorflow import keras
from config import config
import tensorflow as tf
from keras import Model


class MyGan(Model, ABC):
    def __init__(self, input_shape, disc_extra_steps, generator_model, discriminator_model):
        super(MyGan, self).__init__()

        self.shape = input_shape
        self.generator = generator_model
        self.discriminator = discriminator_model

        self.d_optimizer = None
        self.g_optimizer = None
        self.g_loss = []
        self.g_loss_w = []
        self.d_loss = []
        self.d_loss_w = []
        self.disc_extra_steps = disc_extra_steps
        self.d_final_loss_tracker = keras.metrics.Mean(name="d_final_loss")
        self.app_loss_tracker = keras.metrics.Mean(name="app_loss")
        self.advers_loss_tracker = keras.metrics.Mean(name="advers_loss")
        self.g_val_loss = keras.metrics.Mean(name="g_loss")
        self.d_val_loss = keras.metrics.Mean(name="d_loss")

    @property
    def metrics(self):
        return [self.advers_loss_tracker, self.app_loss_tracker, self.d_final_loss_tracker,
                self.g_val_loss, self.d_val_loss]

    def compile(self, **kwargs):
        super(MyGan, self).compile()
        self.d_optimizer = kwargs.get('d_optimizer')
        self.g_optimizer = kwargs.get('g_optimizer')
        self.d_loss = kwargs.get('d_loss')
        self.d_loss_w = kwargs.get('d_loss_w')
        self.g_loss = kwargs.get('g_loss')
        self.g_loss_w = kwargs.get('g_loss_w')
        self.discriminator.trainable = False

    def build(self, input_shape):
        super(MyGan, self).build([input_shape, input_shape])
        self.generator.build(input_shape)
        self.discriminator.build(input_shape)

    @tf.function
    def train_step(self, data):
        d, r = data
        generated_images = self.generator(d)

        fake_labels = tf.zeros(shape=(config.BATCH_SIZE, 1), dtype=tf.float32)
        real_labels = tf.ones(shape=(config.BATCH_SIZE, 1), dtype=tf.float32)

        self.discriminator.trainable = True
        d_final_loss = 0
        with tf.GradientTape(persistent=True) as d_tape:

            for _ in range(self.disc_extra_steps):
                discriminator_real_loss = 0
                discriminator_fake_loss = 0
                disc_predict_real = self.discriminator(r)
                disc_predict_fake = self.discriminator(generated_images)
                d_losses = []
                for i, loss_fn in enumerate(self.d_loss):
                    discriminator_real_loss += loss_fn(real_labels, disc_predict_real)
                    discriminator_fake_loss += loss_fn(fake_labels, disc_predict_fake)
                    d_loss = (discriminator_fake_loss * self.d_loss_w[i]) + (discriminator_real_loss * self.d_loss_w[i])
                    d_losses.append(d_loss)
                d_final_loss += discriminator_real_loss + discriminator_fake_loss
                d_grads = d_tape.gradient(d_losses, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_weights)
                )
            del d_tape

        d_final_loss = (d_final_loss / self.disc_extra_steps) / 2

        self.discriminator.trainable = False

        with tf.GradientTape() as g_tape:
            disc_out, gen_out = self([d, r])
            advers_loss = self.g_loss[0](real_labels, disc_out)
            app_loss = self.g_loss[1](r, gen_out)
            g_grads = g_tape.gradient(
                [(advers_loss * self.g_loss_w[0]), (app_loss * self.g_loss_w[1])], self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(
                zip(g_grads, self.generator.trainable_weights)
            )

        # Update metrics and return their value.
        self.d_final_loss_tracker.update_state(d_final_loss)
        self.app_loss_tracker.update_state(app_loss)
        self.advers_loss_tracker.update_state(advers_loss)
        return {
            "d_final_loss": self.d_final_loss_tracker.result(),
            "advers_loss": self.advers_loss_tracker.result(),
            "app_loss": self.app_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        d, r = data

        generated_images = self.generator.call(d, training=False)

        fake_labels = tf.zeros(shape=(config.BATCH_SIZE, 1), dtype=tf.float32)
        real_labels = tf.ones(shape=(config.BATCH_SIZE, 1), dtype=tf.float32)

        discriminator_real_loss = 0
        discriminator_fake_loss = 0
        disc_predict_real = self.discriminator.call(r, training=False)
        disc_predict_fake = self.discriminator.call(generated_images, training=False)
        for i, loss_fn in enumerate(self.d_loss):
            discriminator_real_loss += loss_fn(real_labels, disc_predict_real)
            discriminator_fake_loss += loss_fn(fake_labels, disc_predict_fake)
        d_final_loss = (discriminator_fake_loss + discriminator_real_loss) / 2

        disc_out, gen_out = self([d, r], training=False)
        app_loss = self.g_loss[1](r, gen_out)

        # Update metrics and return their value.
        self.d_val_loss.update_state(d_final_loss)
        self.g_val_loss.update_state(app_loss)
        return {
            "d_loss": self.d_val_loss.result(),
            "g_loss": self.g_val_loss.result(),
        }

    @tf.function
    def call(self, inputs, **kwargs):
        d, r = inputs
        g_out = self.generator(d)
        d_out = self.discriminator(g_out)
        return d_out, g_out
