'''
author: Pruthviraj Patil
version: 1.0

'''

import numpy as np
import math

class NN:
	def __init__(self, hidden_dims, input_dim):
		# status: incomplete
		self.NN_Params={}

		#network initialization
		self.Network_dims=[{"in_dim": input_dim, "out_dim": hidden_dims[0], "activation":'relu'}]
		out_dim=self.Network_dims[0]["out_dim"]
		for i in range(1, len(hidden_dims)):
			temp={"in_dim":out_dim, "out_dim": hidden_dims[i], "activation":"relu"}
			self.Network_dims.append(temp)
			out_dim=hidden_dims[i]
		self.Network_dims.append({"in_dim":out_dim, "out_dim": 1, "activation":"sigmoid"})

	def print_network(self):
		# status: incomplete
		print("This is the network")
		for index, layer in enumerate(self.Network_dims):
			print("\n Layer: ", index+1)
			print("\t input_dim: ", layer["in_dim"])
			# print("\t Weights: ", self.NN_Params["W_"+str(index+1)])
			# print("\t Bias: ", self.NN_Params["B_"+str(index+1)])
			print("\t activation: ", layer["activation"])
			print("\t output_dim: ", layer["out_dim"], "\n")


	def params_init(self):
		#status: complete
		seed=2132
		np.random.seed(seed)

		for index, layer in enumerate(self.Network_dims):
			self.NN_Params["W_"+str(index+1)]=np.random.randn(layer["out_dim"], layer['in_dim'])*0.01#rows*cols
			self.NN_Params['B_'+str(index+1)]=np.random.randn(layer["out_dim"], 1)*0.02

	def relu(self, y):
		a=np.maximum(0, y)
		return a

	def sigmoid(self, y):
		a=1/(1+np.exp(y))

	def relu_back(self, dA, Z):
		dz=np.array(dA, copy=True)
		dz[Z<=0]=0
		return dz

	def sigmoid_back(self, dA, Z):
		sigmoid=self.sigmoid(Z)
		dz=dA*sigmoid*(1-sigmoid) #derivative


	def forward_pass(self, x, w, b, activation):
		#status: incomplete
		#x= previous activation
		#w= current layer weights
		#b= current layer bias

		y=np.dot(w, x)+b
		if(activation=="relu"):
			return(self.relu(y), y)
		else:
			return(self.sigmoid(y), y)

	def forward_pass_over_network(self, prev_a):
		#status: incomplete
		cache_mem={}
		for index, layer in enumerate(self.Network_dims):
			current_w=self.params_init["W_"+str(index+1)]
			current_b=self.params_init["B_"+str(index+1)]
			if(index==len(Network_dims)-1):
				activation="sigmoid"
			else:
				activation="relu"
			current_a, current_y=self.forward_pass(prev_a, current_w, current_b, activation)
			cache_mem['A_'+str(index+1)]=current_a
			cache_mem['Z_'+str(index+1)]=current_y
			prev_a=current_a
		return (current_a, cache_mem)

	def loss_function(self, y_actual, y):
		#status: incomplete
		m=y_actual.shape[1]
		loss=(-1/m)*(np.dot(y_actual, np.log(y).T)+np.dot((1-y_actual), np.log(1-y).T))
		return np.squeeze(loss)

	def backward_pass(self, current_a, prev_a, current_w, current_b, current_z, activation):
		#status: incomplete
		m= prev_a.shape[1]

		if(activation=="relu"):
			current_z=self.relu_back(current_a, current_z)
		else:
			current_z=self.sigmoid_back(current_a, current_z)
		
		current_w=np.dot(current_z, prev_a.T)/m
		prev_a=np.dot(current_w.T, current_z)
		current_b=np.sum(current_z, axis=1, keepdims=True)/m

		return prev_a, current_w, current_b


	def backward_pass_over_network(self, y, y_actual, cache_mem):
		#status: incomplete
		gradients={}
		y_actual=y_actual.reshape(y)
		prev_da=-(np.divide(y_actual, y)-np.divide(1-y_actual, 1-y))

		for index, layer in reversed(list(enumerate(self.Network_dims))):
			current_a=prev_da
			prev_a=cache_mem["A_"+str(index+1)]
			current_z=cache_mem["Z_"+str(index+1)]
			current_w=self.NN_Params["W_"+str(index+1)]
			current_b=self.NN_Params["B_"+str(index+1)]
			activation=layer["activation"]

			prev_da, current_w, current_b=self.backward_pass(current_a, prev_a, current_w, current_b, current_z, activation)

			gradients["dW_"+str(layer+1)]=current_w
			gradients["dB_"+str(layer+1)]=current_b

		return gradients

	def update(self, gradients, lr):
		for index, layer in enumerate(self.Network_dims):
			self.NN_Params["W_"+str(index+1)]-=lr*gradients["dW_"+str(index+1)]
			self.NN_Params["B_"+str(index+1)]-=lr*gradients["dB_"+str(index+1)]

	def train(self, X, Y, epochs, learning_rate):
		#status: incomplete
		self.params_init()
		costs=[]
		for epoch in range(epochs):
			y_current, cache_mem=self.forward_pass_over_network(X)
			cost=self.loss_function(Y, y_current)
			costs.append(cost)

			gradients=self.backward_pass_over_network(Y, y_current, cache_mem)
			self.update(gradients, learning_rate)

		return self.NN_Params, costs

	def test(self, X):
		# status: incomplete
		a, cache=self.forward_pass_over_network(X)
		last_name=list(cache)[-1]
		y_pred=cache[name]
		return y_pred