import numpy as np
import tensorflow as tf
from tensorflow import keras
from plot import plot_loss
from time_history import TimeHistory, total_time
from matplotlib import pyplot as plt


dataSet = np.load('polution_dataSet.npy')

training = dataSet[0:7000,:]
validate = dataSet[7000:9000,:]
testing = dataSet[9000:11000,:]

timeWindow = 24

train = keras.preprocessing.sequence.TimeseriesGenerator(training, training,
                               length=timeWindow, sampling_rate=1,stride =timeWindow,
                               batch_size=1)

valid = keras.preprocessing.sequence.TimeseriesGenerator(validate, validate,
                               length=timeWindow, sampling_rate=1,stride =timeWindow,
                               batch_size=1)

test = keras.preprocessing.sequence.TimeseriesGenerator(testing, testing,
                               length=timeWindow, sampling_rate=1,stride =timeWindow,
                               batch_size=1)


#GRU4:
#GRU using : loss = mse and optimizer : RMSprop
GRU4 = keras.models.Sequential()
GRU4.add(keras.layers.GRU(4, input_shape=(timeWindow,8 )))
GRU4.add(keras.layers.Dense(8))
GRU4.compile(loss='mse', optimizer=keras.optimizers.RMSprop())


checkpointer = keras.callbacks.ModelCheckpoint(filepath='GRU4_weights.hdf5'
                               , verbose=0
                               , save_best_only=True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss'
                             , patience=10
                             , verbose=0)
GRU4_json = GRU4.to_json()
with open("GRU4.json", "w") as jason_file:
    jason_file.write(GRU4_json)
time_callback = TimeHistory()
GRU4_history = GRU4.fit_generator(train, epochs=100, validation_data=valid
                                              , verbose=0, callbacks= [checkpointer, earlystopper,time_callback])

plot_loss(GRU4_history, 'GRU4 - Train & Validation Loss')

print('time for 10 epochs in seconds:' , total_time(10,time_callback.times))


prediction = GRU4.predict_generator(test)
pr_y = np.zeros((1,len(prediction)))
t_y = np.zeros((1,len(test)))
for num in range(len(prediction)):
    pr_y[:,num]=prediction[num][0]
    t_y[:,num]=(test[num][1])[0][0]

plt.plot(t_y, pr_y, 'ro')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.show()




