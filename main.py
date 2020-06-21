'''
author: Pruthviraj Patil
version: 1.0

'''

from ImageLoader import Img_load
from utilities import image_utility
import cv2

class Image_rec:
	def __init__(self):
		self.img_load=Img_load()
		self.util=image_utility()
#load and preproocess 
#status: complete
	def load_and_preprocess(self, path):
		img_path=path
		images=img_load.load_img(img_path)
		gray_images=util.convert_rgb_gray(images)
		return gray_images

#extract hog (histogram oriented gradients) features
#status: complete
	def get_hog_features(self, images):
		hog_feats=[]
		for image in images:
			hog_feats.append(util.generate_hog(image))
		return hog_feats



#main function to run the model
#status: incomplete
	def run_model(self, paths):

		#load and preprocess the images
		xtrain_neg_path=paths[0]
		xtrain_pos_path=paths[1]
		xtest_neg_path=paths[2]
		xtest_pos_path=paths[3]

		xtrain_pos=self.load_and_preprocess(x_train_pos_path)
		xtrain_neg=self.load_and_preprocess(x_train_neg_path)
		xtest_pos=self.load_and_preprocess(x_test_pos_path)
		xtest_neg=self.load_and_preprocess(x_test_neg_path)

		print("preprocess complete")
		
		#get the hog features
		train_pos_hogs=self.get_hog_features(xtrain_pos)
		train_neg_hogs=self.get_hog_features(xtrain_neg)
		test_pos_hogs=self.get_hog_features(xtrain_pos)
		test_neg_hogs=self.get_hog_features(xtrain_neg)

		print("feature extraction done")

		#pass the hog features to neural network

		

