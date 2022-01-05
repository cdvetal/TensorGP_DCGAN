from tensorgp.engine import *
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

full_fset = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log', 'max', 'mdist', 'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep', 'sstepp', 'step', 'sub', 'tan', 'warp', 'xor'}
extended_fset = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt', 'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}
custom_set = {'sstep', 'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos', 'log', 'warp'}

class GAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channel)

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

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

        # generator takes noise as input and generates imgs

    # Build and compile discriminator
    def DM(self):
        optimizer = Adam(0.0002, 0.5)
        DM = self.build_discriminator()
        DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return DM

class dcgan(object):
    def __init__(self, batch_size = 50, gens_per_batch = 100):
        self.img_rows = 28

        self.img_cols = 28
        self.channels = 1
        self.batch_size = batch_size
        self.gens_per_batch = gens_per_batch
        self.last_gen_imgs = []

        self.GAN = GAN()
        self.DM = self.GAN.DM()

        #call generator
        resolution = [self.img_rows, self.img_cols]

        self.generator = Engine(fitness_func=self.disc_forward_pass,
                                population_size=self.batch_size,
                                tournament_size=3,
                                mutation_rate=0.1,
                                crossover_rate=0.95,
                                max_tree_depth =10,
                                target_dims=resolution,
                                #method='grow',
                                method='ramped half-and-half',
                                objective='maximizing',
                                device='/gpu:0',
                                stop_criteria='generation',
                                function_set=normal_set,
                                min_init_depth=0,
                                max_init_depth=10,
                                min_domain=-1,
                                max_domain=1,
                                bloat_control='dynamic_dep',
                                elitism=0,
                                stop_value=self.gens_per_batch,
                                effective_dims=2,
                                seed=202020212022,
                                debug=0,
                                save_to_file=10, # save all images from each 10 generations
                                save_graphics=True,
                                show_graphics=False,
                                write_gen_stats=False,
                                write_log = False,
                                write_final_pop = True,
                                read_init_pop_from_file = None)

        # training input
        # To change dataset, place dataset below
        (self.x_train, _), (_, _) = mnist.load_data()
        self.x_train = self.x_train / 127.5 - 1.
        self.x_train = np.expand_dims(self.x_train, axis=3)
        # x_train = x_train/127.5 -1.
        # x_train = np.expand_dims(x_train, axis=3)
        self.n_samples = 25
        self.noise_dim = 100

    # maior predict do discriminador -> maior fitness
    def disc_forward_pass(self, **kwargs):
        population = kwargs.get('population')
        generation = kwargs.get('generation')
        tensors = kwargs.get('tensors')
        f_path = kwargs.get('f_path')
        _resolution = kwargs.get('resolution')
        _stf = kwargs.get('stf')

        images = True
        # set objective function according to min/max
        fit = 0
        condition = lambda: (fit > max_fit)  # maximizing
        max_fit = float('-inf')

        fn = f_path + "gen" + str(generation).zfill(5)
        fitness = []
        best_ind = 0
        fit_array = self.DM.predict(np.array(np.expand_dims(tensors, axis = 3)))
        # scores
        for index in range(len(tensors)):
            if generation % _stf == 0:
                save_image(tensors[index], index, fn, _resolution)  # save image
            fit = fit_array[index]

            if condition():
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            population[index]['fitness'] = fit

        # save best indiv
        if images:
            save_image(tensors[best_ind], best_ind, fn, _resolution, addon='_best')
        return population, best_ind

    # method to generate noise
    def gennoise(self, batch_size):
        x = np.random.normal(0, 1.0, (batch_size, self.noise_dim))
        return x

    def plt_imgs(self, epoch):
        self.last_gen_imgs = np.array(self.last_gen_imgs)
        self.last_gen_imgs = 0.5 * self.last_gen_imgs + 0.5  # .... [-1, 1] to [0, 1]

        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(self.last_gen_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1

        fig.savefig("mnist_%d.png" % epoch)
        plt.close()

    def train(self, n_epochs):
        train_hist = {'D_losses': [], 'G_losses': []}

        print("Start")
        true_labels = np.ones((self.batch_size, 1))
        gen_gene_labels = np.zeros((self.batch_size, 1))

        for epoch in range(n_epochs):

            index = np.random.randint(0, self.x_train.shape[0], self.batch_size)
            images = self.x_train[index]

            # train generator
            _, gen_imgs = self.generator.run(self.batch_size)

            # traind disc
            d_loss = self.DM.train_on_batch(images, true_labels)
            d_loss_generated = self.DM.train_on_batch(gen_imgs, gen_gene_labels)
            total_d_loss = 0.5 * np.add(d_loss, d_loss_generated)

            train_hist['D_losses'].append(total_d_loss[0])

            g_loss = -total_d_loss[0]
            #g_loss = self.combined.train_on_batch(noise_data, y1)

            train_hist['G_losses'].append(g_loss)
            print(' Epoch:{}, G_loss: {}, D_loss:{}'.format(epoch + 1, g_loss, total_d_loss[0]))

            if epoch % 50 == 0:
                self.plt_imgs(epoch)

        return train_hist


if __name__ == '__main__':
    mnist_dcgan = dcgan(batch_size=32)
    hist = mnist_dcgan.train(2)