import numpy as np
import itertools
import pandas as pd
import random
import matplotlib.pyplot as plt
import operator



epsilon = 0
alpha = 0.8
gamma = 0.01

def stateGenerator(numofStates):
	states = list(itertools.product(list(range(3)), repeat = numofStates))
	return states

def validMove(s1, s2):
	diff = tuple(abs(x-y) for x,y in zip(s1,s2))
	counter = 0
	for d in diff:
		if not(d == 0):
			counter += 1

	if counter > 1:
		return False
	else:
		for i in range(0, len(diff)):
			if diff[i] == 0:
				continue
			else:
				for j in range(len(s1)):
					if s1[i] == s1[j] and i > j: #(0, 0) -> (0, 1)
						return False
					if s1[j] == s2[i]:
						if j > i:
							return True
						else:
							return False
				return True

	return False

def RMatrix(states, target):
	counter = 0
	R = pd.DataFrame(index=states, columns=states, data=-np.inf)
	for s1 in range(len(states)):
		for s2 in range(len(states)):
			if validMove(states[s1], states[s2]):
				if states[s2] == target:
					R.at[states[s1], states[s2]] = 100
				else:
					R.at[states[s1], states[s2]] = -0.01
	return R

def QMatrix(R, states, target):
	moves = []
	moves2 = []
	Q = pd.DataFrame(index=states, columns=states, data=0)
	episodes = 150
	val = []
	for e in range(episodes):
		v = 0
		move = 0
		state = states[0]
		while not(state == target):
			move += 1
			chance = random.random()
			nextStates = {}
			nextNext = {}
			for s in states:
				if not(R[state][s] == -np.inf):
					nextStates[s] = Q[s][state]
			sortedNextStates = sorted(nextStates.items(), key=operator.itemgetter(1))

			if chance <= 1-epsilon:
				nextState = sortedNextStates[-1][0]
			else:
				index = random.randint(0, len(nextStates)-1)
				nextState = sortedNextStates[index][0]
			for s2 in states:
				if not(R[nextState][s2] == -np.inf):
					nextNext[s2] = Q[s2][nextState]

			sortedNextNext = sorted(nextNext.items(), key=operator.itemgetter(1))

			maxNext = sortedNextNext[-1][0]
			v = sortedNextNext[-1][1]
			newQ = Q.at[state, nextState] + alpha*(R.at[state, nextState] + gamma*v - Q.at[state, nextState])
			Q.ix[state, nextState] = newQ
			state =  nextState
			v += newQ
		# print "**************************************", "Move:", move, "**************************************"
		moves.append(move)
		val.append(v/move)
	fig, ax = plt.subplots()
	ax.plot(range(1, len(val)+1), val)
	ax.set_title('Average of Cumulative Values per Epoch')
	plt.ylabel('Average of Cumulative Values')
	plt.xlabel('epoch')
	plt.show()	

	print Q
	return moves


disks =  int(raw_input("Enter Num of Disks:"))
states = stateGenerator(disks)
if disks == 3:
	target = (2, 2, 2)
elif disks == 4:
	target = (2, 2, 2, 2)
else:
	target = (2, 2, 2, 2, 2)
R = RMatrix(states, target)
# print RMatrix(states, target)
moves = QMatrix(R, states, target)
fig, ax = plt.subplots()
ax.plot(range(1, len(moves)+1), moves)
ax.set_title('number of moves need to achieve goal')
plt.ylabel('move')
plt.xlabel('epoch')
plt.show()