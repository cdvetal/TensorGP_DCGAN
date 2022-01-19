import tensorflow as tf
tf.__version__

from tensorgp.engine import *

#import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

full_fset = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log', 'max', 'mdist',
             'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep', 'sstepp', 'step', 'sub', 'tan',
             'warp', 'xor'}
extended_fset = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt',
                 'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}
#custom_set = {'sstep', 'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos', 'log', 'warp'}
custom_set = {'add', 'cos', 'div', 'if', 'min', 'mult', 'sin', 'sub', 'tan', 'sstepp'}

class dcgan(object):

    def __init__(self,
                 batch_size=32,
                 buffer_size=60000,
                 gens_per_batch=100,
                 run_dir=None,
                 gp_fp=None):

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.input_shape = [self.img_rows, self.img_cols, self.channels]

        date = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-3]
        #print(date)
        self.run_dir = os.getcwd() + delimiter + "gp_dcgan_results" + delimiter + "run__" + date + delimiter if run_dir is None else run_dir
        self.gp_fp = self.run_dir + "gp" + delimiter if gp_fp is None else gp_fp
        self.gan_images = self.run_dir + "dcgan_images" + delimiter

        #os.makedirs(self.run_dir)
        #print("Created dir: ", self.run_dir)

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gens_per_batch = gens_per_batch
        self.last_gen_imgs = []

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator = self.make_discriminator_model()
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        resolution = [self.img_rows, self.img_cols]
        self.generator = Engine(fitness_func=self.disc_forward_pass,
                                population_size=self.batch_size,
                                tournament_size=2,
                                mutation_rate=0.3,
                                crossover_rate=0.8,
                                max_tree_depth=14,
                                target_dims=resolution,
                                # method='grow',
                                method='ramped half-and-half',
                                objective='maximizing',
                                device='/gpu:0',
                                stop_criteria='generation',
                                operators=custom_set,
                                min_init_depth=3,
                                max_init_depth=6,
                                terminal_prob=0.5,
                                min_domain=-1,
                                max_domain=1,
                                bloat_control='std',
                                elitism=1,
                                #stop_value=self.gens_per_batch - 1,
                                stop_value=0,
                                effective_dims=2,
                                seed=202020212022,
                                debug=0,
                                save_to_file=10000,  # save all images from each 10 generations
                                minimal_print=True,
                                save_graphics=True,
                                show_graphics=False,
                                write_gen_stats=False,
                                write_log=False,
                                write_final_pop=True,
                                stats_file_path=self.run_dir,
                                graphics_file_path=self.run_dir,
                                pop_file_path=self.run_dir,
                                run_dir_path=self.gp_fp,
                                read_init_pop_from_file=None,
                                mutation_funcs=[Engine.subtree_mutation, Engine.point_mutation,
                                                Engine.delete_mutation, Engine.insert_mutation],
                                mutation_probs=[0.6, 0.2, 0.1, 0.1]
                                )

        #os.makedirs(self.gp_fp)
        #print("Created dir: ", self.gp_fp)
        os.makedirs(self.gan_images)
        #print("Created dir: ", self.gan_images)

        self.gloss = 0
        self.dloss = 0
        self.training_time = 0
        self.loss_hist = []

        (self.x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.channels).astype('float32')
        self.x_train = (self.x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
        #print(self.x_train.shape)

        #self.train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(self.buffer_size)


    def disc_forward_pass(self, **kwargs):
        population = kwargs.get('population')
        generation = kwargs.get('generation')
        tensors = kwargs.get('tensors')
        _resolution = kwargs.get('resolution')

        fit = 0
        max_fit = float('-inf')

        fitness = []
        best_ind = 0
        # TODO: is predict okay here?
        fit_array = self.discriminator(np.array(np.expand_dims(tensors, axis=3)), training=False)
        # scores
        for index in range(len(tensors)):
            fit = float(fit_array[index][0])

            if fit > max_fit:
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            population[index]['fitness'] = fit

        return population, best_ind


    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.input_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def compute_losses(self, gen_output, real_output):
        gen_loss = self.cross_entropy(tf.zeros_like(gen_output), gen_output)
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        self.dloss = gen_loss + real_loss
        self.gloss = -self.dloss
        self.loss_hist = [self.dloss, self.gloss]

    def print_training_hist(self):
        for h in self.loss_hist:
            print(h)

    def train_step(self, epoch):

        index = np.random.randint(0, self.x_train.shape[0], self.batch_size)
        images = self.x_train[index]

        with tf.GradientTape() as disc_tape:
            #_, generated_images = self.generator.run(self.gens_per_batch)
            _, generated_images = self.generator.run(epoch + 1)
            self.last_gen_imgs = np.expand_dims(generated_images, axis=3)
            gen_output = self.discriminator(self.last_gen_imgs, training=True)
            real_output = self.discriminator(images, training=True)

            self.compute_losses(gen_output, real_output)

            gradients_of_discriminator = disc_tape.gradient(self.dloss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def train(self, epochs=50):
        start = time.time()
        for epoch in range(epochs):
            self.train_step(epoch)

            #for image_batch in self.dataset:

            # Save the model every 15 epochs
            self.generate_and_save_images(epoch + 1)
            if (epoch + 1) % 15 == 0:
                pass
                # checkpoint.save(file_prefix=checkpoint_prefix)

            print('[GAN]:\t[Gloss, Dloss]: [{}, {}]\tTime for epoch {} is {} sec'.format(self.gloss, self.dloss, epoch + 1, time.time() - start))

        # Generate after the final epoch
        # display.clear_output(wait=True)
        self.generate_and_save_images(epochs)
        self.training_time = time.time() - start
        return self.training_time, self.loss_hist


    def generate_and_save_images(self, epoch):
        self.last_gen_imgs = np.array(self.last_gen_imgs)
        self.last_gen_imgs = 0.5 * self.last_gen_imgs + 0.5  # .... [-1, 1] to [0, 1]

        fig = plt.figure(figsize=(8, 4))
        for i in range(self.last_gen_imgs.shape[0]):
            plt.subplot(4, 8, i + 1)
            plt.imshow(self.last_gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.gan_images + 'image_at_epoch_{:04d}.png'.format(epoch))


if __name__ == '__main__':
    epochs = 100
    gens = 50
    gen_pop = 32

    mnist_dcgan = dcgan(batch_size=gen_pop, gens_per_batch=gens)
    train_time, train_hist = mnist_dcgan.train(epochs = epochs)
    print("Elapsed training time (s): ", train_time)
    mnist_dcgan.print_training_hist()
