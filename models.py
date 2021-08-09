import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from resnet import *

# cnn for mnist
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

# vgg16 for cifar10
class Blocks(Sequential):
    def __init__(self,n,m):
        super().__init__()
        for i in range(m):
            self.add(Conv2D(filters = n, kernel_size=(3,3),strides=(1,1),padding = 'same',activation = "relu"))
        self.add(MaxPool2D(pool_size = (2, 2)))

class Denses(Sequential):
    def __init__(self,n,m=2):
        super().__init__()
        for i in range(m):
            self.add(Dense(units = n, activation = "relu"))

class cifar10_vgg11(tf.keras.models.Sequential):
    def __init__(self, input_shape, classes, filters = 64):
        super().__init__()
        self.add(InputLayer(input_shape = input_shape))

        # Backbone
        self.add(Blocks(n = filters * 1, m = 1))
        self.add(Blocks(n = filters * 2, m = 1))
        self.add(Blocks(n = filters * 4, m = 2))
        self.add(Blocks(n = filters * 8, m = 2))
        self.add(Blocks(n = filters * 8, m = 2))

        # top
        self.add(Flatten())
        self.add(Denses(n = filters * 64))
        self.add(Dense(units = classes,activation = "softmax"))


class cifar10_vgg16(Sequential):
    def __init__(self, input_shape, classes, filters = 64):
        super().__init__()
        self.add(InputLayer(input_shape = input_shape))

        # Backbone
        self.add(Blocks(n = filters * 1, m = 2))
        self.add(Blocks(n = filters * 2, m = 2))
        self.add(Blocks(n = filters * 4, m = 3))
        self.add(Blocks(n = filters * 8, m = 3))
        self.add(Blocks(n = filters * 8, m = 3))

        # top
        self.add(Flatten())
        self.add(Denses(n = filters * 64))
        self.add(Dense(units = classes, activation = "softmax"))


        self.add(Dropout(0.1, input_shape=(32, 32, 3)))

        self.add(Conv2D(filters=96, kernel_size=3, activation='relu', padding='same'))  # (32, 32, 96))
        self.add(Conv2D(filters=96, kernel_size=3, activation='relu', padding='same'))  # (32, 32, 96))
        self.add(Conv2D(filters=96, kernel_size=3, strides=2, activation='relu', padding='same'))  # (16, 16, 96)
        self.add(Dropout(0.1))

        self.add(Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'))  # (16, 16, 192))
        self.add(Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'))  # (16, 16, 192))
        self.add(Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same'))  # (8, 8, 192))
        self.add(Dropout(0.1))

        self.add(Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'))  # (8, 8, 192))
        self.add(Conv2D(filters=192, kernel_size=1, activation='relu', padding='same'))  # (8, 8, 192))
        self.add(Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same'))  # (8, 8, 192))

        self.add(Dropout(0.1))
        self.add(Conv2D(filters=10, kernel_size=1, activation='relu', padding='same'))  # (8, 8, 10))

        self.add(AveragePooling2D(pool_size=4, strides=4, padding='valid'))
        self.add(Flatten())
        self.add(Activation('softmax'))

class cifar10_cnn(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Sequential([
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            MaxPooling2D((2, 2))
        ])

        self.conv2 = Sequential([
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            MaxPooling2D((2, 2))
        ])

        self.conv3 = Sequential([
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
            MaxPooling2D((2, 2))
        ])

        self.dense = Sequential([
            Flatten(),
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            Dense(10, activation='softmax')
        ])

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.dense(x)


# simple autoencoder
class ae(Model):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            Flatten(),
            Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
            Dense(np.prod(input_shape), activation='sigmoid'),
            Reshape(input_shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# convolutional autoencoder
class cae(Model):
    def __init__(self, input_shape, latent_dim):
        super().__init__()

        self.encoder = Sequential([
            Input(shape=(input_shape)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
        ])

        self.decoder = Sequential([
            Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            Conv2D(input_shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded