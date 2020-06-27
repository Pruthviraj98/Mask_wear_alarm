'''
author: Pruthviraj Patil
version: 1.0

'''

from ImageLoader import Img_load
from utilities import image_utility
import cv2
import numpy as np
from nnet import net
import random
import urllib.request

# import Tkinter
# from Tkinter import *
# import tkMessageBox
# import ttk

class Image_rec:
	def __init__(self):
		self.img_load=Img_load()
		self.util=image_utility()
		self.neural_net=None
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
		# print("getting hog feats for # imgs: ", len(images))
		eps=0
		for image in images:
			hog_feats.append(self.util.generate_hog(image))
			eps+=1
		# print("but got feats for # of imgs", len(hog_feats))
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

	def real_time_exe(self, url):
		second=0
		while(True):
			imgResponse=urllib.request.urlopen(url)
			imgNP=np.array(bytearray(imgResponse.read()), dtype=np.int8)
			image=cv2.imdecode(imgNP, -1)
			dim=(64, 64)
			img=cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
			gray = 0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]
			grays=[]
			grays.append(gray)

			hogs=self.get_hog_features(grays)
			prep=hogs[0].flatten()
			ips=[]
			ips.append(prep)
			frames=np.array(ips, np.float32)
			# print("image_input=", frames.shape)

			if (cv2.waitKey(1) & 0xFF == ord('q')):
				break

			# window = Tkinter.Tk()
			# window.wm_withdraw()
			# window.geometry("1x1+"+str(window.winfo_screenwidth()/2)+"+"+str(window.winfo_screenheight()/2))

			y_pred=self.neural_net.test(frames)
			if(second%3==0):
				cv2.imshow("ipcap", image)
				if(y_pred[0][1]>0.53):
					print(y_pred, "wear mask")
					# tkMessageBox.showinfo(title="Status", message="Wear Mask Please")
				else:
					print(y_pred, "Thanks for wearing mask")
					# tkMessageBox.showinfo(title="Status", message="Good, you wore the mask")
			second+=1

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


		self.neural_net=net(1, hidden_dims, input_dim)

		x_train=np.array(x_train, np.float32)
		y_train=np.array(y_train, np.int32)

		x_test=np.array(x_test, np.float32)
		print("finalIp", x_test.shape, type(x_test))
		y_test=np.array(y_test, np.int32)

		self.neural_net.train(x_train, y_train, 100, 3, 0.05)
		y_pred=self.neural_net.test(x_test)	
		return y_test, y_pred, train_imgs, test_imgs

model=Image_rec()
tr_n='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/train_negative'
tr_p='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/train_positive'
te_n='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/test_negative'
te_p='C:/Users/pruth/Desktop/reviseCV/sampleProjHD/test_positive'

paths=[tr_n, tr_p, te_n, te_p]

actuals, predictions, train_imgs, test_imgs=model.run_model(paths)

trNo=0
for trains in train_imgs:
	cv2.imwrite("C:/Users/pruth/Desktop/reviseCV/sampleProjHD/train_st/image"+str(trNo)+".jpg", trains)
	trNo+=1
teNo=0
for tests in test_imgs:
	cv2.imwrite("C:/Users/pruth/Desktop/reviseCV/sampleProjHD/test_st/image"+str(teNo)+".jpg", tests)
	teNo+=1

print("actuals=", actuals)
print("predictions=",predictions)



###time for gettin realtime

# url='http://192.168.43.1:8081/shot.jpg' #enter your url here
model.real_time_exe(url)
