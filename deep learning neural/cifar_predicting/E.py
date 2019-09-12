import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/255, x_test/255
numbers_of_each_label = np.zeros(10)
x_train2, y_train2 = x_train, y_train
select_arr = (6000, 32, 32, 3)
select_target = (6000, 1)
y_train_selected = np.zeros(select_target)
x_train_selected = np.zeros(select_arr)
array_select_counter = 0
for i in range(9):
  array_counter = 0
  while True:
    if i == y_train2[array_counter]:
      y_train_selected[array_select_counter] = y_train2[array_counter]
      x_train_selected[array_select_counter, :, :, :] = x_train2[array_counter, :, :, :]
      numbers_of_each_label[i] = numbers_of_each_label[i]+1
      array_select_counter = array_select_counter + 1
    if numbers_of_each_label[i] > 600:
      break
    array_counter = array_counter+1
shuffling = np.arange(6000)
np.random.shuffle(shuffling)
for i in range (5999):
  y_train_selected[i] = y_train_selected[shuffling[i]]
  x_train_selected[i, :, :, :] = x_train_selected[shuffling[i], :, :, :]





#array_counter = 0
#for i in range(50000):
#    random_var = np.random.random_integers(50000-i)
#    target_label = y_train2[random_var]
#    if numbers_of_each_label[target_label] < 101:
#        numbers_of_each_label[target_label] = numbers_of_each_label[target_label]+1
#        x_train_selected[array_counter, :, :, :] = x_train2[random_var, :, :, :]
#        y_train_selected[array_counter] = y_train2[random_var]
#        x_train2 = np.delete(x_train2, slice(random_var, random_var+1), 0)
#        y_train2 = np.delete(y_train2, slice(random_var, random_var+1), 0)
#        array_counter = array_counter+1
#    if numbers_of_each_label[target_label] > 100:
#        x_train2 = np.delete(x_train2, slice(random_var, random_var + 1), 0)
#        y_train2 = np.delete(y_train2, slice(random_var, random_var + 1), 0)
#    if (numbers_of_each_label[0]>100)&(numbers_of_each_label[1]>100)&(numbers_of_each_label[2]>100)&(numbers_of_each_label[3]>100)&(numbers_of_each_label[4]>100)&(numbers_of_each_label[5]>100)&(numbers_of_each_label[6]>100)&(numbers_of_each_label[7]>100)&(numbers_of_each_label[8]>100)&(numbers_of_each_label[9]>100):
#      break

model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(),
  tf.keras.layers.Conv2D(filters=80, kernel_size=6, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)),
  tf.keras.layers.MaxPooling2D(pool_size=2),
  tf.keras.layers.Conv2D(filters=38, kernel_size=4, strides=1, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(210, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(90, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




epok = 10
train_acc = []
test_acc = []
train_loss = []
test_loss = []
train_acc = np.asarray(train_acc)
test_acc = np.asarray(test_acc)
train_loss = np.asarray(train_loss)
test_loss = np.asarray(test_loss)





for i in range(epok):
  history = model.fit(x_train_selected, y_train_selected, batch_size=200, epochs=1)
  score = model.evaluate(x_test, y_test)
  test_acc = np.append(test_acc, [score[1]])





plt.plot(test_acc)
#plt.plot(test_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()