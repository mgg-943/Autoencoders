import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Dense, Layer, Input
from keras.models import Model
from keras import ops

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)  = mnist.load_data()

X_train = x_train.astype('float32') / 255
X_test = x_test.astype('float32') / 255

X_train = X_train.reshape((len(x_train), np.prod(X_test.shape[1:])) )
X_test = X_test.reshape((len(x_test), np.prod(X_test.shape[1:]))) 
print(X_train.shape, X_test.shape)

# Hyperparameters
epoch = 10
batch_size = 100
original_dim = 784
intermediate_dim = 64
latent_dim = 2

# Encoder

encoderIn = Input(shape=(original_dim,), name = "input")
encoder_H1 = Dense(units=intermediate_dim, 
                   activation='relu',
                   name="encoding")(encoderIn)
z_mu = Dense(units=latent_dim,
             name="mean")(encoder_H1)
z_log_sigma2 = Dense(units=latent_dim,
                     name="log_var")(encoder_H1)

class Sampling(Layer):
    def call(self,args):
        z_mu, z_log_sigma2 = args
        eps = keras.random.normal(shape=(ops.shape(z_mu)))
        return z_mu + ops.exp(z_log_sigma2/2)*eps

z = Sampling()([z_mu, z_log_sigma2])
encoder = Model(inputs=encoderIn, outputs=[z_mu, z_log_sigma2, z], name="encoder")

# Decoder

decoderIn = Input(shape=(latent_dim,), name="decoder_input")
decoder_H1 = Dense(intermediate_dim,
                   activation='relu',
                   name="decoder_h")(decoderIn)
decoderOut = Dense(original_dim,
                   activation='sigmoid',
                   name="flat_decoded")(decoder_H1)
decoder = Model(inputs=decoderIn, outputs=decoderOut)


# As an object create the VAE  with parameters (encoder, decoder)

class VAE(Model):
    def __init__(self, encoder, decoder, **kargs):
        super().__init__(**kargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='recontruction loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs):
        """Forward pass for model (called by Keras when not using custom `train_step`)."""
        z_Mean, z_Log_var, z_p = self.encoder(inputs)
        reconstruction = self.decoder(z_p)
        return reconstruction


    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum( keras.losses.binary_crossentropy(reconstruction, data),
                        axis=-1 )
                )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=-1)) 
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    




# Create and train the model 

data = np.concatenate((X_train, X_test), axis=0)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(data,epochs=epoch,batch_size=batch_size,verbose=2)
#vae.fit(X_train,epochs=epoch,batch_size=batch_size,validation_data=(X_test, X_test),verbose=2)


def plot_latent_space(vae, n=15, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    x_0, y_0 = 0,0

    grid_x = np.linspace(-scale + x_0, scale + x_0, n)
    grid_y = np.linspace(-scale + y_0, scale + y_0, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()

plot_latent_space(vae)
