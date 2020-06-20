#installed jedi-python autocompletion package
import cv2
import os
class Img_load:
	def __init__(self):
		self.images=[]


	def load_img(self, loc):
		#status: complete
		img_names=os.listdir(loc)

		for img_name in img_names:
			image=cv2.imread(loc+'/'+img_name)
			self.images.append(image)

		return self.images