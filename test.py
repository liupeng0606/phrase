import pandas as pd

from keras.layers import Dense, Input

from keras.models import Model

from sklearn.metrics import mean_absolute_error


#
data = pd.read_csv("./train_encode/word_m.csv", sep=",", header=None).values
label = pd.read_csv("./train_labels/word_m.csv", sep=",", header=None).values

input = Input(shape=(768,), dtype='float32')

dense1 = Dense(300, activation="tanh")(input)

output = Dense(2)(dense1)
#
model = Model(input, output)
#
model.summary()
#
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#
model.fit(data, label, nb_epoch=20, batch_size=4, validation_split=0.2)

data = pd.read_csv("./test_encode&labels/word_m_encode.csv", sep=",", header=None).values

y = model.predict(data)



y_true = pd.read_csv("./test_encode&labels/word_m_labels.csv", sep=",", header=None).values



y_pred = y

print(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
print(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))


p_data = pd.DataFrame(y)

p_data.to_csv("Y.csv", header=None, index=None)