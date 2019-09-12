from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math



class Node:
	def __init__(self, data, parent):
		self.data = data
		self.parent = parent

	def setAttr(self, attr):
		self.attr = attr

	def  setLabel(self, label):
		self.label = label


class Tree:
	
	def __init__(self, root, attributes, level):
		self.levels = [None]*16
		for i in range(0, 16):
			self.levels[i] = []
		self.buildTree(root, attributes, level)
	def getLevel(self):
		i = 0
		for level in self.levels:
			if level == []:
				continue
			else:
				print "....", len(level)
				i = i + 1
		return i
		
	def buildTree(self, root, attributes, level):
		# print "root", data
		if(level == 16):
			return
		if len(root.data) == 0:
			print "state 1: No more example, majority of parent:", majority(root.parent.data)
			print "________________________________________________________________"
			root.setLabel(majority(root.parent.data))
			self.levels[level].append(root)
			return
		if len(attributes) == 0:
			print np.shape(root.data)
			print root.data[0], "nnnnnn"
			print "state 2: No more attribute; majority of data:", level, majority(root.data)
			print "________________________________________________________________"
			root.setLabel(majority(root.data))
			self.levels[level].append(root)
			return 
		classFreq = {}
		for entry in root.data:
			if (classFreq.has_key(entry[16])):
				classFreq[entry[16]] = classFreq[entry[16]] + 1
			else:
				classFreq[entry[16]] = 1

		for i in range(1, 27):
			if (classFreq.has_key(i)):
				if classFreq[i] == sum(classFreq.values()):
					print "state 3: All examples have same class; class is:", i
					print "________________________________________________________________"
					# return i
					root.setLabel(i)
					self.levels[level].append(root)
					return
			else:
				classFreq[i] = 0
		
		# best = bestAttr(root.data, attributes)
		best = giniBestAttr(root.data, attributes)
		print "best:", best
		print "level:", level
		root.setAttr(best)
		root.setLabel(None)
		self.levels[level].append(root)
		# print attributes
		print "best:", best, "*"
		attributes.remove(best)
		
		children = {}
		for i in range(0, 16):
			children[i] = []

		for entry in root.data:
			children[entry[best]].append(entry)
		successors = [None]*16
		for i in range(0, 16):
			successors[i] = Node(children[i], root)
		# print "children:", children
		# print "_______________________________________________________________________"
		for child in successors:
			child.parent = root
			self.buildTree(child, attributes, level+1)






def entropy(data):
	valFreq = {}
	dataEntropy = 0
	for entry in data:
		if (valFreq.has_key(entry[16])):
			valFreq[entry[16]] = valFreq[entry[16]] + 1
		else:
			valFreq[entry[16]] = 1
	for freq in valFreq.values():
		if freq == 0:
			continue
		else:
			dataEntropy = dataEntropy + (-freq/len(data)) * math.log(freq/len(data), 2) 
	# print "entropy:", dataEntropy
	return dataEntropy			

def IG(data, targetAttr):

	dataSubset = [None]*16
	for i in range(0, 16):
		dataSubset[i] = [entry for entry in data if entry[targetAttr] == i]

	dataEntropy = 0
	subsetEntropy = 0
	
	for val in range(0, 16):
		valProb = len(dataSubset[val]) / len(data)
		subsetEntropy = subsetEntropy + valProb * entropy(dataSubset[val])		
	return (entropy(data) - subsetEntropy)

def gini(data):
	classFreq = {}
	for entry in data:
		if (classFreq.has_key(entry[16])):
			classFreq[entry[16]] = classFreq[entry[16]] + 1
		else:
			classFreq[entry[16]] = 1
	sigma = 0
	for freq in classFreq.values():
		sigma = sigma + (freq/len(data))**2
	return 1-sigma

def giniBestAttr(data, attributes):
	giniIndices = []
	sigma = 0
	for attr in attributes:
		children = {}
		for i in range(0, 16):
			children[i] = []
		for entry in data:
			children[entry[attr]].append(entry)
		for child in children.values():
			sigma = sigma + (len(child)/len(data))*gini(child)
		giniIndices.append(sigma)
	leastCost = np.argmin(giniIndices)
	return attributes[leastCost]

	 

def bestAttr(data, attributes):
	# print attributes, "aaa"
	# print attributes, "(((((((((((((((((("
	n = len(attributes)
	igs = np.zeros((n))
	for i in range(n-1):
		igs[i] = IG(data, attributes[i])
	# print igs,"================================================================="
	return attributes[np.argmax(igs)]
	# print entropy(data)

def majority(data):
	print np.shape(data), "====="
	
	classFreq = np.zeros((28))
	for entry in data:
		if np.shape(entry)[0] == 1:
			entry = entry.T
		print np.shape(entry), "=-=-=-"
		print entry[16]
		classFreq[entry[16]] = classFreq[entry[16]] + 1
	return np.argmax(classFreq)

def test(data, tree):
	prediction  = []
	treeHeight = tree.getLevel()
	for entry in data:
		attr = tree.levels[0][0].attr
		print "first move:", attr
		print entry, "___________________________________"
		for i in range(1, treeHeight):
			attrVal = entry[attr]
			print "attrVal:", attrVal
			branch = tree.levels[i][attrVal]
			if (branch.label == None):
				print "None"
				attr = tree.levels[i][attrVal].attr
				print "attr:", attr
			else:
				prediction.append(branch.label)
				break
	return prediction

def ova(data, labels, attributes):
	fakeLabels = [None]*27
	trees = []
	for i in range(0, 27):
		fakeLabels[i] = []
		for l in labels:
			if l == i:
				fakeLabels[i].append(i)
			else:
				fakeLabels[i].append(0)
	
	for i in range(1, 27):
		print np.shape(np.array(fakeLabels[i]).T), "**", np.shape(data)
		train_data = np.concatenate((data, np.matrix(fakeLabels[i]).T), axis=1)
		# print np.shape(train_data), "====="
		root = Node(train_data, None)
		tree = Tree(root, attributes, 0)
		trees.append(tree)
		for j in range(0, len(data)):
			if(labels[j] == i):
				np.delete(data, data[j])
				np.delete(labels, labels[j])
				for fl in fakeLabels:
					print "hi", fl
					print fl[j]
					fl.remove(fl[j])

	return trees



def confusion(prediction, labels):
	confusionMatrix = np.zeros((27, 27))
	for i in range(1, 27):
		confusionMatrix[labels[i], prediction[i]] =  confusionMatrix[labels[i], prediction[i]] + 1
	return confusionMatrix


# prediction = test(root.data, tree)

def main():
	train_data = scipy.io.loadmat('letter_recognition.mat')['train_data']
	train_labels = [scipy.io.loadmat('letter_recognition.mat')['train_labels'][:, 0]]
	test_data = scipy.io.loadmat('letter_recognition.mat')['test_data']
	test_labels = [scipy.io.loadmat('letter_recognition.mat')['test_labels'][:, 0]]
	classifiedData = {}
	for i in range(1, 27):
		classifiedData[i] = []

	for i in range(len(train_data)):
		classifiedData[train_labels[0][i]].append(train_data[i])
	attributes = range(16)
	tl = np.array(train_labels)
	tt = np.array(test_labels)

	data = np.concatenate((train_data, tl.T), axis=1)
	t_data = np.concatenate((test_data, tt.T), axis=1)
	# data = [[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 1],
	# 		[0, 1, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2],
	# 		[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2],
	# 		[0, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 3],
	# 		[2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 1],
	# 		[1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2]
	# 		]
	# bestAttr(data, attributes)
	# print buildTree(data, attributes, None)
	root = Node(t_data, None)
	tree = Tree(root, attributes, 0)
	print "where???????????????????????????????????"
	prediction = test(root.data, tree)
	print "prediction:", prediction
	labels = t_data[:, 16]
	# labels = []
	# for entry in data:
	# 	labels.append(entry[16])
	
	correctPrediction = 0
	for i in range(0, len(labels)):
		if (prediction[i] == labels[i]):
			correctPrediction = correctPrediction + 1
		else:
			pass
	print "correctPrediction:", correctPrediction
	accuracy = (correctPrediction * 100)/len(labels)
	print "accuracy:", accuracy
	print "_____________________________________________________________________________________________"
	for i in range(0, 3):
		for child in tree.levels[i]:
			print "level:", i
			print "child:", child.data
	print "_____________________________________________________________________________________________"
	print "confusion matrix:"
	print confusion(prediction, labels)
	# print tree.getLevel(), "_-_-"
	
	# ova(train_data, train_labels[0], attributes)
if __name__ == "__main__":
	main()