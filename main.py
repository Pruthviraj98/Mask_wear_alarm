'''
author: Pruthviraj Patil
version: 1.0

'''

from ImageLoader import Img_load
from utilities import image_utility
import cv2
from neural_net import NN

class Image_rec:
	def __init__(self):
		self.img_load=Img_load()
		self.util=image_utility()
#load and preproocess 
#status: complete
	def load_and_preprocess(self, path):
		img_path=path
		images=self.img_load.load_img(img_path)
		gray_images=self.util.convert_rgb_gray(images)
		return gray_images

#extract hog (histogram oriented gradients) features
#status: complete
	def get_hog_features(self, images):
		hog_feats=[]
		for image in images:
			hog_feats.append(self.util.generate_hog(image))
		return hog_feats


	def prep_data_to_train(self, pos_img_feats, neg_img_feats):
		#status: incomplete
		x-[]
		y=[]
		for i in pos_img_feats:
			x.append(i.flatten())
			y.append(1)
		for j in neg_img_feats:
			x.append(i.flatten())
			y.append(0)
		temp=list(zip(x, y))
		np.random.shuffle(temp)
		x, y=zip(*temp)
		x=np.array(x)
		y=np.array(y)

		return x, y




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

		a, b, c=train_pos_hogs[0].shape
		input_dim=a*b*c
		nn=NN([250, 450, 550],  input_dim)
		x_train, y_train=self.prep_data_to_train(train_pos_hogs, train_neg_hogs)
		x_test, y_test=self.prep_data_to_train(test_pos_hogs, test_neg_hogs)

		nn.print_network()

		losses=[]
		for x, y in zip(x_train, y_train):
			params, loss=nn.train(x, y, 100, 0.05)
			losses.append(loss)

		preds=[]
		for x in x_test:
			preds=nn.test(x)
		preds=np.array(preds)
		accuracy=self.util.accuracy(preds, y_test)
		print(accuracy)