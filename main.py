'''
author: Pruthviraj Patil
version: 1.0

'''

from ImageLoader import Img_load
from utilities import image_utility
import cv2
import numpy as np
from nnet import net
import  random

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
		print("getting hog feats for # imgs: ", len(images))
		eps=0
		for image in images:
			hog_feats.append(self.util.generate_hog(image))
			eps+=1
		print("but got feats for # of imgs", len(hog_feats))
		return hog_feats


	def prep_data_to_train(self, pos, neg, pos_img_feats, neg_img_feats):
		#status: complete
		x=[]
		y=[]
		imgs=[]

		for i in range(len(pos_img_feats)):
			x.append(pos_img_feats[i].flatten())
			y.append(1)
			imgs.append(pos[i])

		for j in range(len(neg_img_feats)):
			x.append(neg_img_feats[i].flatten())
			y.append(0)
			imgs.append(neg[i])

		temp=list(zip(x, y, imgs))
		np.random.shuffle(temp)
		x, y, imgs=zip(*temp)

		return x, y, imgs

#main function to run the model
#status: complete
	def run_model(self, paths):

		#load and preprocess the images
		xtrain_neg_path=paths[0]
		xtrain_pos_path=paths[1]
		xtest_neg_path=paths[2]
		xtest_pos_path=paths[3]

		xtrain_pos=self.load_and_preprocess(xtrain_pos_path)
		xtrain_neg=self.load_and_preprocess(xtrain_neg_path)
		xtest_pos=self.load_and_preprocess(xtest_pos_path)
		xtest_neg=self.load_and_preprocess(xtest_neg_path)

		print("preprocess complete")
		
		#get the hog features
		train_pos_hogs=self.get_hog_features(xtrain_pos)
		train_neg_hogs=self.get_hog_features(xtrain_neg)
		test_pos_hogs=self.get_hog_features(xtest_pos)
		test_neg_hogs=self.get_hog_features(xtest_neg)

		print("feature extraction done")

		#pass the hog features to neural network
		hidden_dims=[300, 450, 600]
		a, b, c=train_pos_hogs[0].shape
		input_dim=a*b*c
		x_train, y_train, train_imgs=self.prep_data_to_train(xtrain_pos, xtrain_neg, train_pos_hogs, train_neg_hogs)
		x_test, y_test, test_imgs=self.prep_data_to_train(xtest_pos, xtest_neg, test_pos_hogs, test_neg_hogs)

		neural_net=net(1, hidden_dims, input_dim)

		x_train=np.array(x_train, np.float32)
		y_train=np.array(y_train, np.int32)

		x_test=np.array(x_test, np.float32)
		y_test=np.array(y_test, np.int32)

		# print("shape:",len(train_pos_hogs))
		# print("y's:", y_test, "len:", len(y_test))

		neural_net.train(x_train, y_train, 100, 3, 0.05)
		y_pred=neural_net.test(x_test)	
		return y_test, y_pred, train_imgs, test_imgs

model=Image_rec()
tr_n='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/train_negative'
tr_p='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/train_positive'
te_n='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/test_negative'
te_p='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/test_positive'

paths=[tr_n, tr_p, te_n, te_p]
actuals, predictions, train_imgs, test_imgs=model.run_model(paths)

print("actuals=", actuals)
print("predictions=",predictions)

trNo=0
for trains in train_imgs:
	cv2.imwrite("C:/Users/pruth/Desktop/reviseCV/sampleProjHD/train_st/image"+str(trNo)+".jpg", trains)
	trNo+=1

teNo=0
for tests in test_imgs:
	cv2.imwrite("C:/Users/pruth/Desktop/reviseCV/sampleProjHD/test_st/image"+str(teNo)+".jpg", tests)
	teNo+=1
