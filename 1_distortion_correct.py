import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# TODO 1. Provide an example of a distortion-corrected image.
mtx_dist__pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))
mtx = mtx_dist__pickle["mtx"]
dist = mtx_dist__pickle["dist"]

# distortion-corrected image
image = mpimg.imread('test_images/test1.jpg')
# Undistorting the image:
img_undist = cv2.undistort(image, mtx, dist, None, mtx)
print("Undistorted Finish")

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(img_undist)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
