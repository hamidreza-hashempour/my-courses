from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

TRAIN_SIZE = 60000
TEST_SIZE = 1000
rho = 0.5
lr = 10

def sigmoid(x):
	return 1/(1+np.exp(-x/200))

def sigmoidDrivation(x):
	return x*(1-x)




class NeuralNetwork:
	def __init__(self):
		self.input_size = 784
		self.output_size = 784
		self.hidden_size = 323
		self.b1 = np.random.uniform(low=0, high=1, size=(1, self.hidden_size))
		self.b2 = np.random.uniform(low=0, high=1, size=(1, self.output_size))
		self.W1 = np.random.uniform(low=0, high=1, size=(self.input_size, self.hidden_size))
		self.W2 = np.random.uniform(low=0, high=1, size=(self.hidden_size, self.output_size))



	def feedForward(self, pixels):
		self.first_layer_synapse = np.dot(pixels, self.W1) + self.b1
		self.first_layer_activation = sigmoid(self.first_layer_synapse)
		self.output_layer_synapse = np.dot(self.first_layer_activation, self.W2) + self.b2
		self.output_layer_activation = sigmoid(self.output_layer_synapse)
		return self.output_layer_activation


	def backPropagation(self, pixels, actual_output, predicted_output):
		output_deltas = [0.0] * self.output_size
		output_diff = pixels - predicted_output
		output_deltas = output_diff * sigmoidDrivation(self.output_layer_activation)

		self.dw2 = np.dot(self.first_layer_activation.T, output_deltas)
		self.dw1 = np.dot(self.W2, output_deltas.T)
		self.dw1 = self.dw1.T * sigmoidDrivation(self.first_layer_activation)
		self.dw1 = np.dot(self.dw1.T, pixels) / len(pixels)
		self.W1 += lr * (self.dw1.T + rho*self.dw1.T)
		self.W2 += lr * (self.dw2 + rho*self.dw2)

		print np.shape(pixels)
		errors = [0]*len(pixels)
		for k in range(len(pixels)):
			errors[k] = sum((pixels[k]- predicted_output[k])**2)

		return errors



	def batchTrain (self, train_data, train_labels):
		errors = []
		for i in range(30):
			print ("epoch:"+ str(i)), "_________________________________________________________________________"
			for j in range(50):
				prediction = self.feedForward(train_data[j*200:j*200+200, :])
				error = self.backPropagation(train_data[j*200:j*200+200, :], train_labels[j*200:j*200+200, :], prediction)
				meanError = sum(error)/TRAIN_SIZE
				errors.append(meanError)
				print 'MSE=', meanError


		
			a = self.feedForward(train_data[:30])
			b = a*255
			d = []
		for k in range(30):
			d.append(np.sum(a[k]-train_data[k]))
			# c = b[k].reshape(28, 28)
			# plt.imshow(c)
			# plt.savefig('nn_b_nums2/'+str(k)+'.png')
			# plt.close()
		print "similarity:"
		print d

		return errors


def loadData():

	test_file = np.loadtxt('MNIST/mnist_test.csv', dtype=np.float32, delimiter=',')[:10000,:]
	test_data = test_file[:,1:]
	for i in range(0, 784):
		_max = max(test_data[:, i])
		_min = min(test_data[:, i])
		if (_max - _min) == 0:
			continue
		test_data[:, i] = (test_data[:, i] - _min)/(_max - _min)
	test_labels = test_data
	
	return test_data, test_labels, test_data, test_labels


def main():
	train_data, train_labels, test_data, test_labels = loadData()
	nn = NeuralNetwork()
	errors = nn.batchTrain(test_data, test_data)
	plt.plot(errors)
	plt.show()



if __name__ == "__main__":
	main()