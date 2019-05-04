import pandas as pd
import keras.backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras import metrics
from keras.layers import concatenate
from sklearn.metrics import mean_absolute_error
from keras.callbacks import TensorBoard


batch_size = 4
latent_dim = 8
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
dense1 = Dense(100, activation="tanh")(main_input)
pre_y = Dense(2)(dense1)
con_layer = concatenate([main_input, pre_y], axis=1)

z_mean = Dense(latent_dim)(con_layer)
z_log_var = Dense(latent_dim)(con_layer)
z = Lambda(sampling)([z_mean, z_log_var])

decoder_h = Dense(100, activation='tanh')(z)
x_decoded_mean = Dense(encoding_size)(decoder_h)

def vae_loss(x, x_decoded_mean):
    xent_loss = metrics.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
    return xent_loss+kl_loss
def y_loss(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)


model = Model(inputs=[main_input], outputs=[pre_y, x_decoded_mean])
model.compile(loss=[y_loss, vae_loss], optimizer='adam', metrics=['mae'], loss_weights=[1., 0.8])
model.summary()

test_data = pd.read_csv("./test_encode&labels/word_m_encode.csv", sep=",", header=None).values
test_true = pd.read_csv("./test_encode&labels/word_m_labels.csv", sep=",", header=None).values

model.fit([data],[label, data], batch_size=batch_size,
          epochs=10000,  validation_data=([test_data], [test_true,test_data]),
          callbacks=[TensorBoard(log_dir='/home/liu/keras-log')])

y = model.predict(test_data)
y_pred = y[0]
print(mean_absolute_error(test_true[:, 0], y_pred[:, 0]))
print(mean_absolute_error(test_true[:, 1], y_pred[:, 1]))
