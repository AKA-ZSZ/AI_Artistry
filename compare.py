# import the necessary packages

# Author: Adrian Rosebrock, PhD

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
import matplotlib.pyplot as plt
import numpy as np
import cv2


def compare_images(imageA, imageB, title):
	# compute the normalizes root mean squared error and structural similarity
	# index for the images
	m = nrmse(imageA, imageB)
	s = ssim(imageA, imageB, multichannel=True)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("NRMSE: %.2f, SSIM: %.2f" % (m, s))
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

# load the images set 1
# original_no_HM = cv2.imread("imgs/final_imgs/result_100_1.jpg") # NST - original content and style images, no HM 
# pre_HM = cv2.imread("imgs/final_imgs/result_100_2.jpg") # HM before NST - original content image and altered style image
# post_HM = cv2.imread("imgs/final_imgs/result_100_1_post.jpg") # HM after NST - HM after the result from original NST

# # load the images set 2
# original_no_HM = cv2.imread("imgs/final_imgs/result_100_1_set2.jpg") # NST - original content and style images, no HM 
# pre_HM = cv2.imread("imgs/final_imgs/result_100_2_set2.jpg") # HM before NST - original content image and altered style image
# post_HM = cv2.imread("imgs/final_imgs/result_100_1_post_set2.jpg") # HM after NST - HM after the result from original NST

# load the images set 3
original_no_HM = cv2.imread("imgs/final_imgs/result_100_1_set3.jpg") # NST - original content and style images, no HM 
pre_HM = cv2.imread("imgs/final_imgs/result_100_2_set3.jpg") # HM before NST - original content image and altered style image
post_HM = cv2.imread("imgs/final_imgs/result_100_1_post_set3.jpg") # HM after NST - HM after the result from original NST

# convert the images to RGB
original_no_HM = cv2.cvtColor(original_no_HM, cv2.COLOR_BGR2RGB)
pre_HM = cv2.cvtColor(pre_HM, cv2.COLOR_BGR2RGB)
post_HM = cv2.cvtColor(post_HM, cv2.COLOR_BGR2RGB)

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
