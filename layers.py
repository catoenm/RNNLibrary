
import copy, numpy as np
from helpers import *

class InputLayer(object):
	def __init__(self, input_dim, output_dim):
		print "Initialized Input Layer"
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.synapse = 2*np.random.random((input_dim,output_dim)) - 1
		self.synapse_update = np.zeros_like(self.synapse)


class RNNLayer(object):

	def __init__(self, input_dim, output_dim):
		print "Initialized RNN Layer"

		self.input_dim = input_dim
		self.output_dim = output_dim

		self.synapse = 2*np.random.random((input_dim, output_dim)) - 1
		self.synapse_h = 2*np.random.random((output_dim, output_dim)) - 1

		self.synapse_update = np.zeros_like(self.synapse)
		self.synapse_h_update = np.zeros_like(self.synapse_h)


class OutputLayer(object):
	def __init__(self, dim):
		print "Initialized Output Layer"
		self.dim = dim

def forwardPropRecurrent(input_values, synapse1, output_values, synapse2):
	print "Hello"
	return sigmoid(np.dot(input_values,synapse1) + np.dot(output_values,synapse2))

def forwardPropOutput(input_values, synapse):
	return sigmoid(np.dot(input_values,synapse))

   
