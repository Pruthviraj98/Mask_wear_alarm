'''
author: Pruthviraj Patil
version: 1.0

'''

import numpy as np
import cv2
import Hog 

class image_utility(Hog.hog_feature_generator):
	def __init__(self):
		self.gray_images=[]
		self.hog_feat=hog()
		self.Hog_feats=Hog.hog_feature_generator
		self.neuralnet=neural_net.NN

	def convert_rgb_gray(self, images):
		#status: complete
		for img in images:
			frame = 0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]
			self.gray_images.append(frame)
		return self.gray_images

	
	def generate_hog(self, image):
		#status: incomplete 
		features=self.Hog_feats.generate_feats(self, image, (8, 8), 2)
		return features

	def accuracy(self, y_pred, y):
		sames=np.sum(y_pred, y)
		return(len(y)/sames)
