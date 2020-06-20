import numpy as np
import cv2
from canny import Canny_edge_detector

class hog_feature_generator:
	def __init__(self):
		self.gx=None
		self.gy=None
		self.magnitudes=None
		self.directions=None
		self.canny_util=Canny_edge_detector()
		self.image_into_pos=None

	def generate_grads(self, image):
		#status: complete
		self.gx, self.gy=self.canny_util.grad_from_image(image)

	def generate_directions(self):
		#status: complete
		self.directions=self.canny_util.dir_from_grads(self.gx, self.gy)

	def generate_magnitudes(self):
		#status: complete
		self.magnitudes=self.canny_util.mag_from_grads(self.gx, self.gy)

	def histogram_for_position(self, m_matrix, d_matrix):
		#status: incomplete


	def generate_feats(self, image, hist_window_dims, b_size):
		#status: incomplete
	#steps as said in README

	#step 2 : generate grads, directions, magnitudes
		self.generate_grads(image)
		self.generate_directions()
		self.generate_magnitudes()

	#step 3 : defining positions
		x_dim=hist_window_dims[0]
		y_dim=hist_window_dims[1]
		x, y=self.magnitudes.shape
		
		self.image_into_pos=np.zeros((x/x_dim, y/y_dim, 9))#9 histogram valued array in each position.

		for i in range(x/x_dim):
			for y in range(y/y_dim):
				histogram_array=self.histogram_for_position(self.magnitudes[i*x_dim: i*x_dim+x_dim, j*y_dim: j*y_dim+y_dim], self.directions[i*x_dim: i*x_dim+x_dim, j*y_dim: j*y_dim+y_dim])
				self.image_into_pos[i][j]=histogram_array

		# return (self.magnitudes)
		
image=cv2.imread("sample_fruits.jpg")
image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hog_test=hog_feature_generator()
mags= hog_test.generate_feats(image, 2, 1)

print(mags)
# print(np.argwhere(dire > 180))