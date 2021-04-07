#!/usr/bin/env python
 
'''
Welcome to the Histogram Matching Program!
 
Given a source image and a reference image, this program
returns a modified version of the source image that matches
the histogram of the reference image.
 
Image Requirements:
  - Source image must be color.
  - Reference image must be color.
  - The sizes of the source image and reference image do not
    have to be the same.
  - The program supports an optional third image (mask) as
    an argument.
  - When the mask image is provided, it will be rescaled to
    be the same size as the source image, and the resulting
    matched image will be masked by the mask image.
 
Usage:
  python histogram_matching.py <source_image> <ref_image> [<mask_image>]
'''
 
# Python 2/3 compatibility
from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import os

# Project: Histogram Matching Using OpenCV
# Author: Addison Sears-Collins
# Date created: 9/27/2019
# Python version: 3.7
 
# Define the file name of the images
cwd=os.getcwd()

# SOURCE_IMAGE = f"{cwd}\imgs\src_imgs\starrynight.jpg"

iterations=int(input("Please enter the iterations: ")) # give the iterations of the image
 
SOURCE_IMAGE = f"{cwd}\imgs\\final_imgs\\result_{iterations}_2.jpg" # change iterations here, 1 means the generated image used color profile of original content image

 
def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
    
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_histograms(src_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
 
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
 
    
    # export data into csvs
    write_1D_list_to_csv(f"data/src_hist_blue_{iterations}_2",src_hist_blue)
    write_1D_list_to_csv(f"data/src_hist_green_{iterations}_2",src_hist_green)
    write_1D_list_to_csv(f"data/src_hist_red_{iterations}_2",src_hist_red)

    write_1D_list_to_csv(f"data/src_cdf_blue_{iterations}_2",src_cdf_blue)
    write_1D_list_to_csv(f"data/src_cdf_green_{iterations}_2",src_cdf_green)
    write_1D_list_to_csv(f"data/src_cdf_red_{iterations}_2",src_cdf_red)
    
    # create hists
    # configure and draw the histogram figure of source image
    # plt.figure()
    # plt.title("Color Histogram of Source Image")
    # plt.xlabel("Color value")
    # plt.ylabel("Pixels Count")
    # plt.xlim([0, 256])  # <- named arguments do not work here

    # plt.plot(bin_0[0:-1], src_hist_blue,color='blue')  # <- or here
    # plt.plot(bin_1[0:-1], src_hist_green,color='green')  # <- or here
    # plt.plot(bin_2[0:-1], src_hist_red,color='red')  # <- or here
    # plt.show()

    # # configure and draw the histogram figure of ref image
    # plt.figure()
    # plt.title("Color Histogram of Reference Image")
    # plt.xlabel("Color value")
    # plt.ylabel("Pixels Count")
    # plt.xlim([0, 256])  # <- named arguments do not work here

    # plt.plot(bin_3[0:-1], ref_hist_blue,color='blue')  # <- or here
    # plt.plot(bin_4[0:-1], ref_hist_green,color='green')  # <- or here
    # plt.plot(bin_5[0:-1], ref_hist_red,color='red')  # <- or here
    # plt.show()

def write_1D_list_to_csv(li_header,li):
    np.savetxt(f"{li_header}.csv", li, delimiter=",", fmt='%s', header=li_header)
 
def main():
    """
    Main method of the program.
    """
    start_the_program = input("Press ENTER to calculate the color histogram...") 
 
    # Pull system arguments
    try:
        image_src_name = sys.argv[1]
        
    except:
        image_src_name = SOURCE_IMAGE
        
 
    # Load the images and store them into a variable
    image_src = cv2.imread(cv2.samples.findFile(image_src_name))
 
    # Check if the images loaded properly
    if image_src is None:
        print('Failed to load source image file:', image_src_name)
        sys.exit(1)
    else:
        # Do nothing
        pass
        
    # Calculate the matched image
    calculate_histograms(image_src)
 
    cv2.waitKey(0) # Wait for a keyboard event
 
if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()