from __future__ import print_function, division
import tensorflow
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, concatenate, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

class BIGAN():

    def __init__(self):

        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100



        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        # Build the encoder
        self.encoder = self.build_encoder()
        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False
        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)
        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)
        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])
        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)

    def build_encoder(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))
        model.summary()
        img = Input(shape=self.img_shape)
        z = model(img)
        return Model(img, z)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()



        z = Input(shape=(self.latent_dim,))

        gen_img = model(z)
        return Model(z, gen_img)



    def build_discriminator(self):
        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])
        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)
        

        return Model([z, img], validity)


    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        (X_train, _), (_, _) = cifar10.load_data()
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        gen_loss_vec = np.zeros((epochs, 1))
        disc_loss_vec = np.zeros((epochs, 1))
        disc_acc_vec = np.zeros((epochs, 1))
        disc_loss_fake_vec = np.zeros((epochs, 1))
        disc_acc_fake_vec = np.zeros((epochs, 1))
        disc_loss_real_vec = np.zeros((epochs, 1))
        disc_acc_real_vec = np.zeros((epochs, 1))
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Sample noise and generate img
            z = np.random.normal(size=(batch_size, self.latent_dim))
            imgs_ = self.generator.predict(z)
            # Select a random batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            imgs = np.reshape(imgs, (batch_size,32,32,3))
            z_ = self.encoder.predict(imgs)
            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])
            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            disc_loss_vec[epoch] = d_loss[0]
            disc_acc_vec[epoch] = 100 * d_loss[1]
            disc_loss_fake_vec[epoch] = d_loss_fake[0]
            disc_acc_fake_vec[epoch] = 100 * d_loss_fake[1]
            disc_loss_real_vec[epoch] = d_loss_real[0]
            disc_acc_real_vec[epoch] = 100 * d_loss_real[1]
            gen_loss_vec[epoch] = g_loss[0]
            # If at save interval => save generated image samples

        

            if epoch % sample_interval == 0:
                self.sample_interval(epoch)

        self.plot(epochs, disc_acc_vec, disc_loss_vec, gen_loss_vec, disc_acc_fake_vec, disc_acc_real_vec)

    def plot(self, epochs, disc_acc_vec, disc_loss_vec, gen_loss_vec, acc_fake, acc_real):
        # print(disc_acc_vec.shap)
        plt.title("GAN Loss = binary_crossentropy / Optimizer = Adam/ Dropout")
        ax1 = plt.subplot(212)
        ax1.plot(range(epochs), disc_acc_vec, 'g')
        ax1.set_title("Discriminator Accuracy")
        ax2 = plt.subplot(221)
        ax2.plot(range(epochs), disc_loss_vec, 'g')
        ax2.set_title("Discriminator Loss")
        ax3 = plt.subplot(222)
        ax3.plot(range(epochs), gen_loss_vec, 'r')
        ax3.set_title("Generator Loss")
        # ax1.grid(b=True, which='minor', linestyle='--', color='b')
        # ax2.grid(color='r', linestyle='-', linewidth=1)
        # ax3.grid(color='r', linestyle='-', linewidth=1)
        plt.show()
        plt.figure(2)
        ax4 = plt.subplot(211)
        ax4.plot(range(epochs), acc_fake, 'g')
        ax4.set_title("Discriminator Accuracy For Fake Digits")
        ax5 = plt.subplot(212)
        ax5.plot(range(epochs), acc_real, 'g')
        ax5.set_title("Discriminator Accuracy For Real Digits")
        plt.show()


    def sample_interval(self, epoch):
        r, c = 5, 5
        z = np.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.generator.predict(z)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('mnist_%d' % epoch)
        plt.close()

if __name__ == '__main__':
    bigan = BIGAN()
    bigan.train(epochs=501, batch_size=32, sample_interval=100)