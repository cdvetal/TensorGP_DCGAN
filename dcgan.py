from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
import numpy as np


class GAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channel)

        optimizer = Adam(0.0002, 0.5)

    def build_discriminator(self):
        model = Sequential()
        depth = 32
        dropout = 0.25
        input_shape = (self.img_rows, self.img_cols, self.channel)

        model.add(Conv2D(depth * 1, 3, strides=2, input_shape=input_shape, padding='same',
                         kernel_initializer='random_uniform'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        model.add(Conv2D(depth * 2, 3, strides=2, padding='same', kernel_initializer='random_uniform'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        model.add(Conv2D(depth * 4, 3, strides=2, padding='same', kernel_initializer='random_uniform'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))
        model.add(Conv2D(depth * 8, 3, strides=2, padding='same', kernel_initializer='random_uniform'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        # Each MNIST input = 28 X 28 X 1, depth = 1
        # Each Output = 14 X 14 X 1, depth = 64
        # Model has 4 convolutional layer, each with a dropout layer in between

        # Output
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()

        img = Input(shape=(self.img_shape))
        validity = model(img)

        return Model(img, validity)

        # generator takes noise as input and generates imgs

    def build_generator(self):
        generator = Sequential()
        dropout = 0.4
        depth = 128
        dim = 7

        # In: 100
        # Out: dim X dim X depth

        generator.add(Dense(dim * dim * depth, input_dim=100))
        generator.add(Activation('relu'))
        generator.add(Reshape((dim, dim, depth)))
        generator.add(UpSampling2D())
        # generator.add(Dropout(dropout))

        # In: dim X dim X depth
        # Out: 2*dim X 2*dim X depth/2

        generator.add(Conv2D(depth, 3, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Activation('relu'))
        generator.add(UpSampling2D())
        generator.add(Conv2D(int(depth / 2), 3, padding='same'))
        generator.add(BatchNormalization(momentum=0.9))
        generator.add(Activation('relu'))

        # Out : 28 X 28 X 1 grayscale image [0.0, 1.0] per pix
        generator.add(Conv2D(1, 3, padding='same'))
        generator.add(Activation('tanh'))
        generator.summary()

        noise = Input(shape=(100,))
        img = generator(noise)

        return Model(noise, img)

    # Build and compile discriminator
    def DM(self):
        optimizer = Adam(0.0002, 0.5)
        DM = self.build_discriminator()
        DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return DM


class dcgan(object):
    def __init__(self):
        self.img_rows = 28

        self.img_cols = 28
        self.channels = 1

        # building the generator
        self.GAN = GAN()
        self.DM = self.GAN.DM()
        self.generator = self.GAN.build_generator()

        z = Input(shape=(100,))
        img = self.generator(z)
        self.DM.trainable = False
        valid = self.DM(img)

        self.combined = Model(z, valid)
        optimizer = Adam(0.0002, 0.5)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        # training input
        # To change dataset, place dataset below
        (self.x_train, _), (_, _) = mnist.load_data()
        self.x_train = self.x_train / 127.5 - 1.
        self.x_train = np.expand_dims(self.x_train, axis=3)
        # x_train = x_train/127.5 -1.
        # x_train = np.expand_dims(x_train, axis=3)
        self.n_samples = 25
        self.noise_dim = 100

    # method to generate noise
    def gennoise(self, batch_size, noise_dim):
        x = np.random.normal(0, 1.0, (batch_size, self.noise_dim))
        return x

    def plt_imgs(self, epoch):
        noise = self.gennoise(self.n_samples, self.noise_dim)

        fake_imgs = self.generator.predict(noise)
        fake_imgs = 0.5 * fake_imgs + 0.5

        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(fake_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1

        fig.savefig("mnist_%d.png" % epoch)
        plt.close()

    def train(self, n_epochs, batch_size):
        train_hist = {}

        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        print("Start")
        true_labels = np.ones((batch_size, 1))
        gen_gene_labels = np.zeros((batch_size, 1))

        for epoch in range(n_epochs):

            index = np.random.randint(0, self.x_train.shape[0], batch_size)
            images = self.x_train[index]

            noise_data = self.gennoise(batch_size, 100)
            gen_imgs = self.generator.predict(noise_data)
            print("gen imgs shape: ", gen_imgs)

            d_loss = self.DM.train_on_batch(images, true_labels)

            d_loss_generated = self.DM.train_on_batch(gen_imgs, gen_gene_labels)

            total_d_loss = 0.5 * np.add(d_loss, d_loss_generated)

            train_hist['D_losses'].append(total_d_loss[0])

            noise_data = self.gennoise(batch_size, 100)
            y1 = np.ones((batch_size, 1))

            g_loss = self.combined.train_on_batch(noise_data, y1)

            train_hist['G_losses'].append(g_loss)
            print(' Epoch:{}, G_loss: {}, D_loss:{}'.format(epoch + 1, g_loss, total_d_loss[0]))

            if epoch % 50 == 0:
                self.plt_imgs(epoch)

        return train_hist


def plotting_imgs(self, epoch):
    noise = self.gennoise(25, 100)
    fake_imgs = self.generator.predict(noise)
    fake_imgs = 0.5 * fake_imgs + 0.5

    fig, axs = plt.subplots(5, 5)
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(fake_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1


if __name__ == '__main__':
    mnist_dcgan = dcgan()
    train_hist = mnist_dcgan.train(1, batch_size=32)