from tensorgp.engine_tf import *

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import PIL
from heapq import nsmallest, nlargest
from keras import layers
import time
from keras.models import load_model
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import sys


delimiter = os.path.sep

gen_image_cnt = 0
fake_image_cnt = 0

# function sets available
full_set = {'abs', 'add', 'and', 'clip', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'lerp', 'log', 'max', 'mdist',
            'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt', 'sstep', 'sstepp', 'step', 'sub', 'tan',
            'warp', 'xor'}
extended_set = {'max', 'min', 'abs', 'add', 'and', 'or', 'mult', 'sub', 'xor', 'neg', 'cos', 'sin', 'tan', 'sqrt',
                'div', 'exp', 'log', 'warp'}
simple_set = {'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos'}
normal_set = {'add', 'mult', 'sub', 'div', 'cos', 'sin', 'tan', 'abs', 'sign', 'pow'}
# custom_set = {'sstep', 'add', 'sub', 'mult', 'div', 'sin', 'tan', 'cos', 'log', 'warp'}
custom_set = {'add', 'cos', 'div', 'if', 'min', 'mult', 'sin', 'sub', 'tan', 'warp'}
#Function set +, −,  * , /, min, max, abs, neg, warp, sign, sqrt, pow, mdist, sin, cos, if
std_set = {'add', 'sub', 'mult', 'div', 'sin', 'cos', 'min', 'max', 'abs', 'neg', 'warp', 'sign', 'sqrt', 'pow', 'mdist', 'if'}

cnn_model = load_model('MNIST_keras_CNN.h5')
dpi = 96


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        D_HIDDEN = 64
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(1, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        """
        # normal
        nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1,1), padding_mode='zeros', bias=False), # change channels to 3 for color
        nn.LeakyReLU(0.2, inplace=True),
        # downsample
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # downsample
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # downsample
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding_mode='zeros', bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # classifier
        #nn.Flatten(),
        #nn.Dropout(p=0.5),
        #nn.Linear()
        nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding_mode='zeros', bias=False),
        nn.Sigmoid()
        """

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class dcgan(object):

    def __init__(self,
                 batch_size=32,
                 gens_per_batch=100,
                 archive_size = 100,
                 archive_stf = 1,
                 starchive = 1,
                 do_archive = False,
                 classes_to_train=None,
                 run_from_last_pop=True,
                 linear_gens_per_batch=False,
                 log_losses=True,
                 seed=202020212022,
                 log_digits_class=True,
                 sufix=None,
                 fset=None,
                 run_dir=None,
                 gp_fp=None,
                 archive_dir=None):

        self.seed = seed
        torch.manual_seed(seed)

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.input_shape = [self.img_rows, self.img_cols, self.channels]

        self.do_archive = do_archive
        self.archive = []
        self.starchive = starchive
        self.archive_size = archive_size
        self.archive_stf = archive_stf
        self.log_losses = log_losses
        self.log_digits_class = log_digits_class

        # print(date)

        self.run_from_last_pop = run_from_last_pop
        self.linear_gens_per_batch = linear_gens_per_batch

        # os.makedirs(self.run_dir)
        # print("Created dir: ", self.run_dir)

        self.batch_size = batch_size
        self.gens_per_batch = gens_per_batch
        self.last_gen_imgs = []

        resolution = [self.img_rows, self.img_cols]
        self.fset = normal_set if fset is None else fset
        stop_value = self.gens_per_batch - 1 if self.linear_gens_per_batch else 4


        #torch stuff
        torch.manual_seed(seed)
        CUDA = torch.cuda.is_available()
        if CUDA: torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda:0" if CUDA else "cpu")
        self.discriminator = Discriminator().to(self.device)
        self.discriminator.apply(self.weights_init)
        print("Discriminator network: ", self.discriminator)
        self.cross_entropy = nn.BCELoss()
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


        self.generator = Engine(fitness_func=self.disc_forward_pass,
                                population_size=self.batch_size,
                                tournament_size=2,
                                mutation_rate=0.3,
                                crossover_rate=0.8,
                                max_tree_depth=14,
                                target_dims=resolution,
                                method='ramped half-and-half',
                                objective='maximizing',
                                device='cpu',
                                stop_criteria='generation',
                                domain_mode='log',
                                operators=self.fset,
                                min_init_depth=3,
                                max_init_depth=6,
                                terminal_prob=0.5,
                                domain=[-1, 1],
                                bloat_control='off',
                                elitism=1,
                                codomain=[-1, 1],
                                do_final_transform = False,
                                stop_value=stop_value,
                                effective_dims=2,
                                seed=self.seed,
                                debug=0,
                                #gen_display_step=10,
                                minimal_print=True, # True

                                # saves
                                save_to_file=1,  # save all images from each 10 generations
                                save_graphics=True,
                                show_graphics=False,
                                exp_prefix='pref',
                                save_image_pop=True,
                                save_image_best=True,
                                image_extension="jpg",
                                save_log=True,
                                save_to_file_log=1,

                                #stats_file_path=self.gp_fp,
                                #graphics_file_path=self.run_dir,
                                #run_dir_path=self.gp_fp,

                                read_init_pop_from_file=None,
                                best_overall_dir=True,
                                mutation_funcs=[Engine.subtree_mutation, Engine.point_mutation,
                                                Engine.delete_mutation, Engine.insert_mutation],
                                mutation_probs=[0.6, 0.2, 0.1, 0.1]
                                )

        # paths
        self.run_dir = self.generator.get_working_dir()
        self.gp_fp = self.run_dir + "gp" + delimiter if gp_fp is None else gp_fp
        self.gan_images = self.run_dir + "dcgan_images" + delimiter
        self.archive_dir = self.run_dir + "archive" + delimiter if archive_dir is None else archive_dir
        self.gallery_res = [1024, 1024]
        self.best_im_dir = self.gan_images + delimiter + "images"

        os.makedirs(self.gan_images)
        if self.do_archive:
            os.makedirs(self.archive_dir)
        os.makedirs(self.best_im_dir)

        self.gloss = 0
        self.dloss = 0
        self.training_time = 0
        self.loss_hist = []

        # sieve classes
        self.classes_to_train = classes_to_train if classes_to_train is not None else [i for i in range(10)]

        self.dataset = dset.MNIST(root='./data', download=True,
                             transform=transforms.Compose([
                                 #transforms.Resize(64), # is resizing
                                 transforms.ToTensor(),
                                 transforms.Normalize((127.5,), (127.5,)) # normalize to -1, 1
                             ]))
        train_mask = np.isin(self.dataset.targets, self.classes_to_train)
        #indices = dataset.targets == 5  # if you want to keep images with the label 5
        self.dataset.data, self.dataset.targets = self.dataset.data[train_mask], self.dataset.targets[train_mask]
        self.dataset.data, self.dataset.targets = self.dataset.data[:64], self.dataset.targets[:64]  # testing purposes
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)




        """
        (self.x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        print(y_train)
        print(self.x_train.shape)
        print(y_train.shape)
        train_mask = np.isin(y_train, self.classes_to_train)
        self.x_train = self.x_train[train_mask]
        #self.x_train = self.x_train[:64] # testing purposes

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.channels).astype(
            'float32')
        self.x_train = (self.x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
        print("Len of selected dataset: ", len(self.x_train))
        self.x_train = tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(len(self.x_train)).batch(self.batch_size)
        #print(self.x_train.shape)
        """

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def disc_forward_pass(self, **kwargs):
        population = kwargs.get('population')
        #generation = kwargs.get('generation')
        #tensors = kwargs.get('tensors')
        _resolution = kwargs.get('resolution')

        fit = 0
        max_fit = float('-inf')

        fitness = []
        best_ind = 0
        tensors = [p['tensor'] for p in population]

        #fit_array = self.discriminator(np.array(np.expand_dims(tensors, axis=self.last_axis)), training=False)
        #fit_array = self.discriminator(np.array(np.expand_dims(tensors, axis=2)), training=False)

        fit_array = self.discriminator(torch.from_numpy(np.expand_dims(tensors, axis=1)).type(torch.FloatTensor)).view(-1)

        # scores
        for index in range(len(tensors)):
            fit = float(fit_array[index].item())

            if fit > max_fit:
                max_fit = fit
                best_ind = index
            fitness.append(fit)
            population[index]['fitness'] = fit

        return population, best_ind



    def compute_losses(self, gen_output, real_output):
        gen_loss = self.cross_entropy(torch.zeros_like(gen_output), gen_output)
        real_loss = self.cross_entropy(torch.ones_like(real_output), real_output)
        gen_loss.backward()
        real_loss.backward()
        self.dloss = gen_loss + real_loss
        self.gloss = -self.dloss
        self.loss_hist.append([self.dloss.item(), self.gloss.item()])

    def print_training_hist(self):
        for h in self.loss_hist:
            print(h)

    def train_step(self, images, step):

        #index = np.random.randint(0, self.x_train.shape[0], self.batch_size)
        #images = self.x_train[index]
        global gen_image_cnt, fake_image_cnt
        fake_image_cnt += len(images)

        ep = self.gens_per_batch if self.linear_gens_per_batch else round(step / 10) + 5
        starchive = self.starchive if step == 0 else 0
        gen_image_cnt += self.batch_size * ep
        #print("Startb form last pop: ", self.run_from_last_pop)
        _, generated_images = self.generator.run(stop_value=ep,
                                                start_from_last_pop=self.run_from_last_pop,
                                                #start_from_archive = starchive,
                                                #archive = self.archive
                                                )


        # rollling archive
        if self.do_archive:
            get_pop = [copy.deepcopy(p) for p in self.generator.population]
            self.archive = nlargest(min(self.archive_size, self.generator.population_size + len(self.archive)), self.archive + get_pop, key=itemgetter('fitness'))

        # tf.debugging.assert_greater_equal(generated_images, -1.0, message="Less than min domain!")
        # tf.debugging.assert_less_equal(generated_images, 1.0, message="Grater than max domain!")

        self.last_gen_imgs = np.expand_dims(generated_images, axis=3)
        classify_digits(self.last_gen_imgs)

        #gen_output = self.discriminator(self.last_gen_imgs, training=True)
        #real_output = self.discriminator(images, training=True)
        #real_output = tf.sigmoid(real_output).numpy()

        # Train discriminator
        # (1) Update the discriminator with real data
        self.discriminator.zero_grad()
        # Format batch
        real_cpu = images.to(self.device)
        real_output = self.discriminator(real_cpu).view(-1)
        # (2) Update the discriminator with fake data
        reshape_ims = np.transpose(self.last_gen_imgs, axes=[0,3,1,2])
        #print(reshape_ims.shape)
        gen_output = self.discriminator(torch.from_numpy(reshape_ims).type(torch.FloatTensor)).view(-1)
        # (3) compute losses
        self.compute_losses(gen_output, real_output)
        # Update Discriminator
        self.disc_optimizer.step()


    def train(self, epochs = 1):
        start = time.time()

        for epoch in range(epochs):
            step = 0
            #for images in self.x_train:
            for i, images in enumerate(self.dataloader, 0):
                self.train_step(images[0], step)
                if self.log_losses: self.write_losses_epochs(step, epoch)
                if self.log_digits_class:
                    self.write_digits_classifications(step, epoch, self.last_gen_imgs)

                # for image_batch in self.dataset:

                self.generate_and_save_images(step + 1, epoch + 1)
                step += 1

                print('[DCGAN - step {}/{} of epoch {}/{}]:\t[Gloss, Dloss]: [{}, {}]\tTime so far: {} sec'.format(step, len(self.dataset),
                                                                                                                   epoch + 1, epochs, self.gloss,
                                                                                                                   self.dloss, time.time() - start))
            # Generate after the final epoch
            self.generate_and_save_images(step + 1, epoch + 1)

            if self.do_archive and ((epoch + 1) % self.archive_stf) == 0:
                print("Saving archive...")
                self.write_archive(epoch)

        self.training_time = time.time() - start
        #self.plot_losses()
        return self.training_time, self.loss_hist


    def print_archive(self):
        for p in self.archive:
            print(p['fitness'])


    def generate_and_save_images(self, s, e):
        self.last_gen_imgs = np.array(self.last_gen_imgs)
        #self.last_gen_imgs = 0.5 * self.last_gen_imgs + 0.5  # .... [-1, 1] to [0, 1]

        fig = plt.figure(figsize=(8, 4))
        for i in range(self.last_gen_imgs.shape[0]):
            plt.subplot(4, 8, i + 1)

            if i == 0:
                tens = self.generator.best['tensor']
                best_tens = self.generator.domain_mapping(tens) #

            plt.imshow(self.last_gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.gan_images + 'image_at_epoch{:04d}_step{:04d}.png'.format(e, s))
        plt.close()

        fig = plt.figure(frameon=False)
        dpi = 96
        fig.set_size_inches(self.gallery_res[0]/dpi, self.gallery_res[1]/dpi)
        plt.imshow(best_tens, cmap='gray')
        plt.axis('off')
        plt.savefig(self.best_im_dir + delimiter + "best_in_batch_{:04d}_step{:04d}.png".format(e, s), dpi=dpi)
        plt.close()


    def write_losses_epochs(self, step, epoch):
        fn = self.run_dir + "dcgan_losses.txt"
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if epoch == 0 and step == 0:
                file.write("[d_loss, g_loss]\n")
            fwriter.writerow([self.dloss.item(), self.gloss.item()])


    def write_digits_classifications(self, step, epoch, digits, classifications = True, path = None):
        if path is None:
            path = self.run_dir
        fn = path + "digit_max.txt"
        header = epoch == 0 and step == 0

        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            if header:
                file.write("[step, epoch, max]\n")
            fwriter.writerow([step, epoch] + list(np.argmax(classify_digits(digits), axis=1)))

        if classifications:
            fn = path + "digit_classifications.txt"
            with open(fn, mode='a', newline='') as file:
                fwriter = csv.writer(file, delimiter=',')
                if header:
                    file.write("[step, epoch, classifications]\n")
                fwriter.writerow([step, epoch] + [list(p) for p in classify_digits(digits).numpy()])


    def plot_losses(self, show_graphics = False):
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(self.loss_hist)), np.asarray(self.loss_hist)[:, 0], linestyle='-', label="D loss")
        pylab.legend(loc='upper left')
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Loss')
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_title('Discriminator loss across training steps')
        fig.set_size_inches(12, 8)
        plt.savefig(fname=self.run_dir + 'Losses.svg', format="svg")
        if show_graphics: plt.show()
        plt.close(fig)

    def write_archive(self, epoch, save_class=True):
        fn = self.archive_dir + delimiter + "epoch_" + str(epoch).zfill(4) + delimiter
        os.makedirs(fn)
        with open(fn + "expressions.txt", mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            file.write("[indiv, fitness, expression]\n")
            c = 0
            for ind in self.archive:
                fwriter.writerow([str(c), ind['fitness'], ind['tree'].get_str()])
                save_image(ind['tensor'], c, fn, self.generator.target_dims, sufix='_archive_best')
                c += 1
        with open(fn + "tensors.txt", mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            file.write("[indiv, expression]\n")
            c = 0
            for ind in self.archive:
                fwriter.writerow([str(c), ind['tensor'].numpy()])
                c += 1
        if save_class:
            archive_digits = np.expand_dims([p['tensor'] for p in self.archive], axis=3)
            self.write_digits_classifications(0, 0, archive_digits, classifications = True, path = fn)



def classify_from_name(imname='test_im.png', invert=True):
    x = io.imread(imname)
    # compute a bit-wise inversion so black becomes white and vice versa
    if invert:
        np.invert(x)
    x = rgb2gray(x)
    # make it the right size
    x = resize(x, (28, 28))
    # print(x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    x = x.astype('float32')
    classify_digits(x)


def classify_digits(digits):
    return cnn_model(digits, training=False)
    #return


    #print(out.shape)
    #print("Output:", out)
    #print("Argmax: ", np.argmax(out, axis=1))


if __name__ == '__main__':

    gen_pop = 32
    #if len(sys.argv) > 1:
    #    print("Going for digit: ", sys.argv[1])
    #    digits = [int(sys.argv[1])]

    # main test for all
    #gens = [50] # 50 # 1- teste maluco (tem de ser pelo menos 2)
    #epochs = 5
    #fsets = [std_set]
    #runs = 30 # 15
    #digits = range(1, 10) # 10



    # secondary test
    gens = [5] # 50 # 1- teste maluco (tem de ser pelo menos 2)
    epochs = 2
    fsets = [std_set]
    runs = 1
    digits = [0]

    seeds = [random.randint(0, 0x7fffffff) for i in range(runs)]
    #seeds = [202020212022]
    #cnn_model.summary()

    for r in range(runs): #jncor podia ser for seed in seeds:
        for d in digits:
            for g in gens:
                for cur_set in fsets:
                    print("doing: ",r," digit ",d," for ", g, " generations, seed ", seeds[r])
                    sufix_str = 'digits_' + str(d) + "_base"
                    mnist_dcgan = dcgan(batch_size=gen_pop, gens_per_batch=g, fset=cur_set, classes_to_train=d,
                                        run_from_last_pop=True,
                                        linear_gens_per_batch=True,
                                        do_archive=False,
                                        starchive=0,
                                        sufix=sufix_str,
                                        seed = seeds[r],
                                        log_losses = True,
                                        log_digits_class = True)
                    train_time, train_hist = mnist_dcgan.train(epochs = epochs)
                    print("Elapsed training time (s): ", train_time)
                    #mnist_dcgan.print_training_hist()
    print("Number of gen image: ", gen_image_cnt)
    print("Number of fake images: ", fake_image_cnt)

    """
    epochs = 100
    gen_pop = 32
    #run_from_last_pop = True
    #linear_gens_per_batch = True
    gens = 100
    fsets = extended_set
    print("\n\nCurrent number of gens: ", gens)
    print("Current set: ", str(fsets))
    print("CRun from last pop?: ", False)
    print("Linear gens per batch?: ", True)
    mnist_dcgan = dcgan(batch_size=gen_pop, gens_per_batch=100, fset=fsets,
                    run_from_last_pop=False, linear_gens_per_batch=True)
    train_time, train_hist = mnist_dcgan.train(epochs = epochs)
    print("Elapsed training time (s): ", train_time)
    mnist_dcgan.print_training_hist()
    """
