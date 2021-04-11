import matplotlib.pyplot as plt
import cv2
import os

def display(img_100,img_300,img_500):
    # initialize the figure
    fig = plt.figure("Images")
    images = ("Pre-Processed Image(100 iterations)", img_100), ("Pre-Processed Image(300 iterations)", img_300), ("Pre-Processed Image(500 iterations)", img_500)
    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap = plt.cm.gray)
        plt.axis("off")
    # show the figure
    plt.show()

img_100=cv2.imread("imgs/final_imgs/result_100_2.jpg") 
img_300=cv2.imread("imgs/final_imgs/result_300_2.jpg") 
img_500=cv2.imread("imgs/final_imgs/result_500_2.jpg") 

img_100 = cv2.cvtColor(img_100, cv2.COLOR_BGR2RGB)
img_300 = cv2.cvtColor(img_300, cv2.COLOR_BGR2RGB)
img_500 = cv2.cvtColor(img_500, cv2.COLOR_BGR2RGB)

display(img_100,img_300,img_500)