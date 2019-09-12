import numpy as np
import matplotlib.pyplot as plt
import math

TRAIN_SIZE = 60000
TEST_SIZE = 10000
lr = 1

# def sigmoid(x):
# 	return 1/(1+np.exp(-x))

# def sigmoidDrivation(x):
# 	return x*(1-x)

#Linear(with fake name :D to reduce changes.)
def sigmoid(x):
	return x

def sigmoidDrivation(x):
	return np.ones((np.shape(x)[0], np.shape(x)[1]))

#tanh(with fake name :D to reduce changes.)
# def sigmoid(x):
# 	return np.tanh(x)

# def sigmoidDrivation(x):
# 	return 1 - np.tanh(x)**2

class NeuralNetwork:
	def __init__(self):
		self.input_size = 784
		self.output_size = 10
		# self.output_size = 4
		self.hidden_size = 64
		self.b1 = np.random.randn(1, self.hidden_size)
		self.b2 = np.random.randn(1, self.output_size)
		self.W1 = np.random.randn(self.input_size, self.hidden_size) 
		self.W2 = np.random.randn(self.hidden_size, self.output_size) 
		# self.W1 = np.zeros((self.input_size, self.hidden_size))
		# self.W2 = np.zeros((self.hidden_size, self.output_size))
		# self.W1 = np.ones((self.input_size, self.hidden_size))
		# self.W2 = np.ones((self.hidden_size, self.output_size))


	def feedForward(self, pixels):
		self.first_layer_synapse = np.dot(pixels, self.W1) + self.b1
		self.first_layer_activation = sigmoid(self.first_layer_synapse)
		self.output_layer_synapse = np.dot(self.first_layer_activation, self.W2) + self.b2
		self.output_layer_activation = sigmoid(self.output_layer_synapse)
		return self.output_layer_activation

	def backPropagation(self, pixels, actual_output, predicted_output):

		output_deltas = [0.0] * self.output_size
		input = np.zeros((784, 1))
		for i in range(len(pixels)):
			input[i] = pixels[i]
		output_diff = actual_output - predicted_output

		output_deltas =  output_diff * sigmoidDrivation(self.output_layer_activation)

		self.dw2 = np.dot(self.first_layer_activation.T, output_deltas)

		self.dw1 = np.dot(self.W2, output_deltas.T)

		self.dw1 = self.dw1.T * sigmoidDrivation(self.first_layer_activation)
		self.dw1 = np.dot(input, self.dw1)

		
		# self.db2 = np.mean(output_diff, axis=0)
		# print self.dw2
		self.W1 += lr * self.dw1
		self.W2 += lr * self.dw2
		# self.b1 -= lr * self.db1
		# self.b2 -= lr * self.db2

		errors = [0]*len(pixels)
		print len(pixels)
		for k in range(len(output_diff)):
			# print np.shape(actual_output), np.shape(predicted_output)
			errors[k] = 0.5 * sum((actual_output[k]- predicted_output.T[k])**2)
		return errors



	def batchTrain (self, train_data, train_labels):
		errors = []
		for i in range(25):
			error = 0
			print ("epoch:"+ str(i)), "_________________________________________________________________________"
			for j in range(TRAIN_SIZE):
				prediction = self.feedForward(train_data[j])
				error += sum(self.backPropagation(train_data[j], train_labels[j], prediction))
			meanError = error/TRAIN_SIZE
			errors.append(meanError)
			print 'MSE=', meanError
		return errors

	def test(self, test_data, test_labels):
		prediction = self.feedForward(test_data)
		accuracy = np.sum(test_labels.argmax(1) == prediction.argmax(1))
		# accuracy = 0
		# for i in range(len(test_labels)):
		# 	accuracy += np.sum(list2bin(test_labels[i]) == math.ceil(list2bin(prediction[i])))
		print "accuracy:", accuracy/100,"%"


def int2bin(i):
	if i == 0: 
		return "0000"
	s = ''
	while i:
		if int(i) & 1 == 1:
			s = "1" + s
		else:
			s = "0" + s
		i /= 2
	return s

def list2bin(x):
	num = 0
	for i in range(4):
		num = x[3-i] * pow(2, i) + num
	return num


def loadData():
	train_file = np.loadtxt('MNIST/mnist_train.csv', dtype=np.float32, delimiter=',')
	train_data = train_file[:,1:]

	for i in range(0, 784):
		_max = max(train_data[:, i])
		_min = min(train_data[:, i])
		if (_max - _min) == 0:
			continue
		train_data[:, i] = (train_data[:, i] - _min)/(_max - _min)
	train_labels1 = train_file[:,0]
	train_labels = np.zeros((TRAIN_SIZE, 10))
	train_labels_bd = np.zeros((TRAIN_SIZE, 4))
	for i in range(TRAIN_SIZE):
		train_labels[i, int(train_labels1[i])] = 1
		#binary decode
		# label = int2bin(train_labels1[i])
		# n = len(label)
		# labels = list(int2bin(train_labels1[i]))[n-4: n]
		# for j in range(len(labels)-1):
		# 	train_labels_bd[i][j] = labels[j]

	test_file = np.loadtxt('MNIST/mnist_test.csv', dtype=np.float32, delimiter=',')
	test_data = test_file[:,1:]
	for i in range(0, 784):
		_max = max(test_data[:, i])
		_min = min(test_data[:, i])
		if (_max - _min) == 0:
			continue
		test_data[:, i] = (test_data[:, i] - _min)/(_max - _min)
	test_labels1 = test_file[:,0]
	test_labels = np.zeros((TEST_SIZE, 10))
	test_labels_bd = np.zeros((TEST_SIZE, 4))
	for i in range(TEST_SIZE):
		test_labels[i, int(test_labels1[i])] = 1
		#binary decode
		# label = int2bin(train_labels1[i])
		# n = len(label)
		# labels = list(int2bin(test_labels1[i]))[n-4: n]
		# for j in range(len(labels)-1):
		# 	test_labels_bd[i][j] = labels[j]
	
	return train_data, train_labels, test_data, test_labels
	# return train_data, train_labels_bd, test_data, test_labels_bd


def main():
	train_data, train_labels, test_data, test_labels = loadData()
	nn = NeuralNetwork()
	errors = nn.batchTrain(train_data, train_labels)
	plt.plot(errors)
	plt.show()
	nn.test(test_data, test_labels)

if __name__ == "__main__":
	main()