# Mask_wear_alarm

The model detects if the person is wearing mask or not in real time (cam input).

## What's so special about this project? - 
Everything (including nn) is designed only using 3 libraries ONLY from scratch- cv2, numpy, math. 

Process Explained:

	1. load image batches
	
	2. convert images to grayscale
	
	3. extract hog features from those images
	
	4. train the features on nn

## Sub program specifications and explainations 

### Hog (Histogram Oriented Gradient) feature extraction features:

	HOG is a feature descriptor. It converts images of size [width, height, 3(channels)] into a feature vector or a array of length n. In case of HOG, distribution of directions of gradients (x, y derivatives) are used as features. This is useful because the magnitude of gradients in greater in edges and corners and that's where the intensity of light changes abruptly and edges pack lots of info about the object.
	Hog features are calculated on a patch of image (of any size). Conventionally, multiple such patches (of different size) at multiple locations of the same image are considered for extracting the feature. 
	So, the steps are: 
	1. (Preprocessing)- selecting the patches from the image. 
	
	2. Calculating the gradient Images
		calculate dx (i.e. gx), dy (i.e. gy), magnitude (i.e. sqrt(gx**2+gy**2)),
		direction (i.e. argtan(gy/gx))
	Here, gx and gy can be calculated by convoluting the patch with the filters like gaussian mask/prewitt  mask.
	
	(note: x-grad= fires on vertical lines
		  y-grad= fires on horizontal lines
		  magnitude= where there is sharp change in intensity
		  For color images, the gradients of the three channels are evaluated ang the max of the gradient pixel among three at every pixel is chosen)
		  
		  
	3. Calculate Histogram of Gradients in 8*8 cells
		First the image is converted to 8*8 cells. Further the histogram of gradients is calculated for each of those cells
		One may ask why 8*8? Answer= 8*8 size helps us gaining more interesting features (face, top of head. etc) and also to gain a compact representation from HOG 			feature descriptor. In an 8*8 patch, there are 8*8*3 pixels (192). The gradient of this pack contains 2 values. i.e. magnitude and direction. So, per pixel in a 		 channel, no. of values = 8*8*2 =128 values. These 128 numbers are represented in -bin histogram that inturn can be stored as an array of 9 numbers. This also 			makes model more robust to noise. The direction of arrows points to the direction of change in intensity and the magnitude shows how big the difference is. 			
		(note: unsigned gradients: angle is in range 0-180 (negatives (>180) are clipped to 180))
	 	
		So, here, histogram of gradients in 8*8 cells are prepared i.e. we create the 9 bins corresponding to angles 0, 20, 40, ....160. The grad direction (angle) pixel 		  values are compared with gradeint magnitude values and are split into bins. A bin is selected based on the direction, and the vote ( the value that goes into the 		    bin ) is selected based on the magnitude.   


	 4. 16*16 Block Normalization
	 	The gradients of the images are sensiive to the change in light intensity. Ex: if we change the pixel value by half, the gradient too decreases by half inturn decreasing the histogram value by half. Thus we need the normalization process to eradicate this sensitivity. For example: Considering the RGB value of  a pixel: [128, 64, 32], length= sqrt((128)**2 + (64)**2 + (32)**2) is 146.32. dividing th vector by this value, gives [0.87, 0.43, 0.22]. Considering another vector, twice of first vector [256, 128, 64], by finding the normalized vector of it, we get the same [0.97, 0.43, 0.22] unchanged. For one block, there are 9*1 histogram, so for 4 blocks, i.e. for 16*16 block, we have 9*4=36 histogram values. Normalizing that, we have normalized 36*1 vector. Further, we move the window further (refer the link :https://www.learnopencv.com/wp-content/uploads/2016/12/hog-16x16-block-normalization.gif). 

	5. Calculating the HOG feature vector
		Here, all the 36*1 vectors are concatenated to form a bigger vector.
		Ex: How many different positions did the 16*16 window moved in the picture?
		(from the link there were 7hor*15vert).
		so totally 7*15=105 positions.
		Hence, 36*1*105=3780 valued vector conatenated to be returned as a hog vector.
		
### Neural Network building explained in breif:

	Step 1. Initialization of the network layers
		The weights and bias are initiated in the beginning for each layer. Weights is the matrix of [n(l), n(l-1)] dimension whereas Bias is the matrix of [n(l), 1]. where l is a layer. 
		Initialize the layers with their respective input, output, activation functions in a dictionary. The input and output vectors  are assigned with the small random values initially.

	Step 2. Define activation functions - relu and 
		sigmoid used in this case. (with an input x)

	Step 3. Define the forward function for single layer 
		taking in the previous activation A(l-1) and calculating current Z. Further calculate the current activation A by passing it through the required function. Return both Z and A. Z is needed for backward propagation further.

	Step 4. Define the fully forward propagation module
		Using the function for single layered forward propagation, define the fully forward propagation module by iterating through the layers and passing on the activations A and getting back Z value. Store A's and Z's in the memory dictionary and return it.

	Step 5. Define Loss function
		Binary Cross entropy in this case. 

	Step 6. Define Backward prop function for single layer
		Refer link to the equations: https://miro.medium.com/max/268/1*FZ4slpsaH_U0YYhaSRqUEQ.gif

	Step 7. Define the fully backward propagation module
		Using the  Single Backward prop function create the fully backward prop module

	Step 8. Create Train, Test and Update modules.
