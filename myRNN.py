import copy, numpy as np
from layers import *
from helpers import *

np.random.seed(1)

binaryValues = {}
maxBits = 8

#Create Map of Integers to binary
largest_number = pow(2,maxBits)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

for i in range(largest_number):
    binaryValues[i] = binary[i]

alpha = 0.1
inputDim = 2
hiddenDim = 8
hidden2Dim = 8
#hidden3Dim = 16
#hidden4Dim = 16
outputDim = 1

Layer0 = InputLayer(2, 8)
Layer1 = RNNLayer(8, 8)
Layer2 = RNNLayer(8, 8)

#Layer Weight Initialization
synapse_0 = 2*np.random.random((inputDim,hiddenDim)) - 1
synapse_1 = 2*np.random.random((hiddenDim,hidden2Dim)) - 1
synapse_h = 2*np.random.random((hiddenDim,hiddenDim)) - 1
synapse2_1 = 2*np.random.random((hidden2Dim,outputDim)) - 1
synapse2_h = 2*np.random.random((hidden2Dim,hidden2Dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
synapse2_1_update = np.zeros_like(synapse2_1)
synapse2_h_update = np.zeros_like(synapse2_h)

#Start Training
for i in range(20000):
	inputInt1 = np.random.randint(largest_number/2) # int version
	inputBin1 = binaryValues[inputInt1]

	inputInt2 = np.random.randint(largest_number/2) # int version
	inputBin2 = binaryValues[inputInt2]

	outputInt = inputInt1 + inputInt2
	outputBin = binaryValues[outputInt]

	netOutput = np.zeros_like(outputBin)

	totalError = 0

	layer_1_values = list()
	layer_1_values.append(np.zeros(hiddenDim))
	layer_2_values = list()
	layer_2_values.append(np.zeros(hiddenDim))
	layer_3_deltas = list()

	for position in range(maxBits):

		X = np.array([[inputBin1[maxBits - position - 1],inputBin2[maxBits - position - 1]]])
		y = np.array([[outputBin[maxBits - position - 1]]]).T

		layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
		layer_2 = sigmoid(np.dot(layer_1,synapse_1) + np.dot(layer_2_values[-1],synapse2_h))
		layer_3 = sigmoid(np.dot(layer_2,synapse2_1))

		layer_3_error = y - layer_3
		layer_3_deltas.append((layer_3_error)*sigmoid_deriv(layer_3))
		totalError += np.abs(layer_3_error[0])
		netOutput[maxBits - position - 1] = np.round(layer_3[0][0])

		layer_1_values.append(copy.deepcopy(layer_1))
		layer_2_values.append(copy.deepcopy(layer_2))

	future_layer_1_delta = np.zeros(hiddenDim)
	future_layer_2_delta = np.zeros(hidden2Dim)

	for position in range(maxBits):
		X = np.array([[inputBin1[position],inputBin2[position]]])
		layer_1 = layer_1_values[-position-1]
		prev_layer_1 = layer_1_values[-position-2]

		layer_2 = layer_2_values[-position-1]
		prev_layer_2 = layer_2_values[-position-2]

		# error at output layer
		layer_3_delta = layer_3_deltas[-position-1]

		layer_2_delta = (future_layer_2_delta.dot(synapse2_h.T) + layer_3_delta.dot(synapse2_1.T)) * sigmoid_deriv(layer_2)

		# error at hidden layer
		layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_deriv(layer_1)


		# let's update all our weights so we can try again
		synapse2_1_update += np.atleast_2d(layer_2).T.dot(layer_3_delta)
		synapse2_h_update += np.atleast_2d(prev_layer_2).T.dot(layer_2_delta)
		synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
		synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
		synapse_0_update += X.T.dot(layer_1_delta)

		future_layer_1_delta = layer_1_delta

	synapse_0 += synapse_0_update * alpha
	synapse_1 += synapse_1_update * alpha
	synapse_h += synapse_h_update * alpha
	synapse2_1 += synapse2_1_update * alpha
	synapse2_h += synapse2_h_update * alpha      

	synapse_0_update *= 0
	synapse_1_update *= 0
	synapse_h_update *= 0
	synapse2_1_update *= 0
	synapse2_h_update *= 0

	if(i % 1000 == 0):
		print "Error:" + str(totalError)
		print "Pred:" + str(netOutput)
		print "True:" + str(outputBin)
		out = 0
		for index,x in enumerate(reversed(netOutput)):
		    out += x*pow(2,index)
		print str(inputInt1) + " + " + str(inputInt2) + " = " + str(out)
		print "------------"

print synapse_0










