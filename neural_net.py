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

	def forward_pass_over_network(self, )


nn=NN([4, 3, 3, 1], 2)
nn.params_init()
nn.print_network()
