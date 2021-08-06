import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *

# simple cnn for mnist
class mnist_cnn(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# simple autoencoder
class ae(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            Flatten(),
            Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
            Dense(784, activation='sigmoid'),
            Reshape((28, 28, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded