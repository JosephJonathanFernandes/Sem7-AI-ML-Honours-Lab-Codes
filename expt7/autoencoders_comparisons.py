import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

# Load data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape((len(x_train), -1)) / 255.
x_test = x_test.reshape((len(x_test), -1)) / 255.

# ---------- 1️⃣ Basic Autoencoder ----------
ae = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid')
])
ae.compile(optimizer='adam', loss='mse')
ae.fit(x_train, x_train, epochs=5, batch_size=256, verbose=0)
ae_loss = ae.evaluate(x_test, x_test, verbose=0)

# ---------- 2️⃣ Denoising Autoencoder ----------
noise_factor = 0.4
x_train_noisy = np.clip(x_train + noise_factor*np.random.normal(size=x_train.shape), 0., 1.)
x_test_noisy = np.clip(x_test + noise_factor*np.random.normal(size=x_test.shape), 0., 1.)
dae = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(784, activation='sigmoid')
])
dae.compile(optimizer='adam', loss='mse')
dae.fit(x_train_noisy, x_train, epochs=5, batch_size=256, verbose=0)
dae_loss = dae.evaluate(x_test_noisy, x_test, verbose=0)

# ---------- 3️⃣ Variational Autoencoder ----------
from tensorflow.keras import layers, models

latent_dim = 2
inputs = layers.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(decoder_input)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(784, activation='sigmoid')(x)
decoder = models.Model(decoder_input, outputs)

vae_outputs = decoder(z)
vae = models.Model(inputs, vae_outputs)
reconstruction_loss = binary_crossentropy(inputs, vae_outputs) * 784
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae.add_loss(K.mean(reconstruction_loss + kl_loss))
vae.compile(optimizer='adam')
vae.fit(x_train, None, epochs=5, batch_size=128, verbose=0)
vae_loss = np.mean(vae.evaluate(x_test, None, verbose=0))

# ---------- Comparison Table ----------
print("\nReconstruction Loss Comparison:")
print(f"Basic Autoencoder Loss:     {ae_loss:.4f}")
print(f"Denoising Autoencoder Loss: {dae_loss:.4f}")
print(f"Variational Autoencoder Loss: {vae_loss:.4f}")
