import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Convolution2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data(path='data')
	number = 10000
	x_train = x_train[0:number]
	y_train = y_train[0:number]
	x_train = x_train.reshape(number, 1, 28,  28)
	x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)
	x_train = x_train
	x_test = x_test


	# x_test = np.random.normal(x_test)   
	# VERY IMPORTANT !!!!!!!!!
	x_train = x_train / 255
	x_test = x_test / 255

	# setting noise, then use dropout to improve performance 
	# x_test = np.random.normal(x_test)

	return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()


# x_train = x_train.reshape(-1, 1, 28, 28)/255
# x_test = x_test.reshape(-1, 1, 28, 28)/255
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)




model = Sequential()

# 1 * 28 * 28
# model.add(Conv2D(25, kernel_size=(3, 3), input_shape = (1, 28, 28), activation='relu')) # 25 * 26 * 26
# model.add(MaxPooling2D((2, 2))) # 25 * 13 * 13
# model.add(Conv2D(50, (3, 3), activation='relu')) # 50 * 11 * 11
# model.add(MaxPooling2D((2, 2))) # 50 * 5 * 5
# model.add(Flatten()) # 1250
# model.add(Dense(units=10, activation='softmax'))


model.add(Convolution2D(filters=25, kernel_size=3, strides=1,padding='same', batch_input_shape=(None, 1, 28, 28), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = 2, strides=2,padding='same',data_format='channels_first'))
model.add(Convolution2D(50, 3, strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same'))

model.add(Flatten())
model.add(Dense(666, activation='relu'))
model.add(Dense(10, activation='softmax'))


# mean square error
# model.compile(loss='mse', optimizer = SGD(lr=0.1), metrics = ['accuracy'])

# categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# batch_size more large, performance more poor
model.fit(x_train, y_train, batch_size = 100, epochs = 20)

result = model.evaluate(x_train, y_train, batch_size = 100000)
print ('Train accuracy: ', result[1])

result = model.evaluate(x_test, y_test, batch_size = 100000)
print ('Test accuracy: ', result[1])


