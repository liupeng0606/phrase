from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class SGAN:
    def __init__(self):
        self.noise_dim = 64
        self.regression_dim = 2
        self.data_dim = 768

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'mse'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['mae']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.noise_dim,))
        gen_data = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        valid, _ = self.discriminator(gen_data)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_generator(self):
        input = Input(shape=(self.noise_dim,))
        gen_data_h = Dense(1000,activation="relu")(input)
        gen_data_h1 = Dense(100, activation="relu")(gen_data_h)
        gen_data = Dense(self.data_dim)(gen_data_h1)
        return Model(input, gen_data)

    def build_discriminator(self):
        input = Input(shape=(self.data_dim,))
        h_dense = Dense(1000,activation="relu")(input)
        h_dense1 = Dense(100, activation="relu")(h_dense)
        valid = Dense(1, activation="sigmoid")(h_dense1)
        regression_val = Dense(self.regression_dim)(h_dense1)
        return Model(input, [valid, regression_val])



    def train(self, train_data, epochs, batch_size=32, sample_interval=50):

        # Get the dataset
        (X_train, y_train) = train_data

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of train_data
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            rel_data = X_train[idx]

            # Sample noise and generate a batch of new data
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            gen_data = self.generator.predict(noise)

            # reg-label
            labels = y_train[idx]
            fake_labels = np.full((batch_size, 2), 5.0)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(rel_data, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_data, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(noise, valid)
            # Plot the progress
            print (d_loss)
            print(g_loss)
            # If at save interval => save model
            if epoch % sample_interval == 0:
                self.save_model()


    def save_model(self):
        def save(model, model_name):
            model.save("./saved_model_word/%s.hdf5" % model_name)
        save(self.generator, "sgan_generator")
        save(self.discriminator, "sgan_discriminator")
        save(self.combined, "sgan_adversarial")
        print("saved!")



data = pd.read_csv("./train_encode/word_encode.csv", sep=",", header=None).values
label = pd.read_csv("./train_labels/word.csv", sep=",", header=None).values
train_data = (data, label)


test_data = pd.read_csv("./test_encode&labels/word_encode.csv", sep=",", header=None).values
test_true = pd.read_csv("./test_encode&labels/word_label.csv", sep=",", header=None).values



# sgan = SGAN()
# sgan.train(train_data,epochs=200000, batch_size=64, sample_interval=50)



from keras.models import load_model

sgan_discriminator = load_model('./saved_model_word/sgan_discriminator.hdf5')


r = sgan_discriminator.predict(test_data)

p_data = pd.DataFrame(r[1])

p_data.to_csv('./result_test/r.csv', sep=',', header=None, index=None)



a = r[1][:, 0]
v = r[1][:, 1]


a_true = test_true[:, 0]
v_true = test_true[:, 1]



print("a pcc", pearsonr(a, a_true))
print("v pcc", pearsonr(v, v_true))
