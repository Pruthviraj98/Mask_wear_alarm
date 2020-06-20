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
#status: incomplete
	def get_hog_features(self, images):
		hog_feats=[]
		for image in images:
			hog_feats.append(util.generate_hog(image))
		return(hog_feats)
	