import pandas as pd
import keras.backend as K
from keras.layers import Dense, Input, Lambda, Layer
from keras.models import Model
from keras import metrics
from keras.layers import concatenate
from sklearn.metrics import mean_absolute_error
from keras.callbacks import TensorBoard

batch_size = 2
latent_dim = 2
epsilon_std = 1.
encoding_size = 768
labeled_ratio = 0.2

data = pd.read_csv("./train_encode/word_m.csv", sep=",", header=None).values
label = pd.read_csv("./train_labels/word_m.csv", sep=",", header=None).values

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma / 2) * epsilon


main_input = Input(shape=(encoding_size,), dtype='float32')
main_out = Input(shape=(latent_dim,), dtype='float32')

dense1 = Dense(100, activation="tanh")(main_input)

z_mean = Dense(latent_dim)(dense1)
z_log_var = Dense(latent_dim)(dense1)
z = Lambda(sampling)([z_mean, z_log_var])


dense2 = Dense(100, activation='tanh')(z)
x_decoded_mean = Dense(encoding_size)(dense2)


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
    def vae_loss(self, x, x_decoded_mean, y, z_mean):
        xent_loss =  metrics.mean_squared_error(x, x_decoded_mean)#Square Loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(y - z_mean) - K.exp(z_log_var), axis=-1)# KL-Divergence Loss
        return xent_loss  + kl_loss
    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        y = inputs[2]
        z_mean = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, y, z_mean)
        self.add_loss(loss, inputs=inputs)
        return z_mean

out = CustomVariationalLayer()([main_input, x_decoded_mean, main_out, z_mean])

model = Model(input=[main_input, main_out], output=out)
model.compile(loss=None, optimizer='adam', metrics=['mae'])

model.summary()

test_data = pd.read_csv("./test_encode&labels/word_m_encode.csv", sep=",", header=None).values
test_true = pd.read_csv("./test_encode&labels/word_m_labels.csv", sep=",", header=None).values

model.fit([data, label], batch_size=batch_size,
          epochs=200, callbacks=[TensorBoard(log_dir='/home/liu/keras-log')])

y = model.predict([test_data,test_true])

print(y.shape)

y_pred = y

print(mean_absolute_error(test_true[:, 0], y_pred[:, 0]))
print(mean_absolute_error(test_true[:, 1], y_pred[:, 1]))
