# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB, multichannel=True)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()

# load the images
original_no_HM = cv2.imread("imgs/final_imgs/result_100_1.jpg") # NST - original content and style images, no HM 
pre_HM = cv2.imread("imgs/final_imgs/result_100_2.jpg") # HM before NST - original content image and altered style image
post_HM = cv2.imread("imgs/final_imgs/result_100_1_post.jpg") # HM after NST - HM after the result from original NST
# convert the images to grayscale
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
# shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
images = ("Original (only NST)", original_no_HM), ("Pre-Processed (HM before NST)", pre_HM), ("Post-Processed (HM after NST)", post_HM)
# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
# show the figure
plt.show()
# compare the images

if __name__ == '__main__':
	compare_images(original_no_HM, original_no_HM, "Original (only NST) vs. Original (only NST)")
	compare_images(original_no_HM, pre_HM, "Original (only NST) vs. Pre-Processed (HM before NST)")
	compare_images(original_no_HM, post_HM, "Original (only NST) vs. Post-Processed (HM after NST)")
