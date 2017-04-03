
import copy, numpy as np


def sigmoid(x):
	value = 1/(1+np.exp(-x))
	return value

def sigmoid_deriv(x):
	value = x*(1-x);
	return value