import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# TODO 2. Use color transforms, gradients, etc., to create a thresholded binary image.
# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img_gray, orint='x', sobel_kernel=3, abs_thresh=(0, 255)):
    if 'x' == orint:
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif 'y' == orint:
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('illegal direction!')
        return
    abs_sobel = np.absolute(sobel)
    scale_sobel = np.uint8(abs_sobel * 255 / np.max(abs_sobel))
    img_binary = np.zeros_like(img_gray)
    img_binary[(scale_sobel >= abs_thresh[0]) & (scale_sobel <= abs_thresh[1])] = 1
    return img_binary


# Define a function that applies Sobel x and y, then computes the
# magnitude of the gradient and applies a threshold
def mag_thresh(img_gray, sobel_kernel=3, mag_thresh=(0, 255)):
    xsobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ysobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(xsobel * xsobel + ysobel * ysobel)
    factor = np.max(mag_sobel)
    mag_sobel = (mag_sobel * 255 / factor).astype(np.uint8)
    img_binary = np.zeros_like(img_gray)
    img_binary[(mag_sobel >= mag_thresh[0]) & (mag_sobel <= mag_thresh[1])] = 1
    return img_binary


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
def dir_thresh(img_gray, sobel_kernel=3, dir_thresh=(0, np.pi / 2)):
    xsobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ysobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel = np.arctan2(np.absolute(ysobel), np.absolute(xsobel))
    img_binary = np.zeros_like(img_gray)
    img_binary[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1
    return img_binary


# Define a function that thresholds the channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_channel(img_origin, channel='h', thresh=(0, 255)):
    hls = cv2.cvtColor(img_origin, cv2.COLOR_RGB2HLS)
    if 'h' == channel:
        img_channel = hls[:, :, 0]
    elif 'l' == channel:
        img_channel = hls[:, :, 1]
    elif 's' == channel:
        img_channel = hls[:, :, 2]
    else:
        print('illegal image channel!')
        return
    # Threshold color channel
    img_binary = np.zeros_like(img_channel)
    img_binary[(img_channel > thresh[0]) & (img_channel <= thresh[1])] = 1
    return img_channel, img_binary


def combinations(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    xsobel_binary = abs_sobel_thresh(img_gray, 'x', 3, (40, 100))
    ysobel_binary = abs_sobel_thresh(img_gray, 'y', 3, (20, 120))
    mag_binary = mag_thresh(img_gray, 3, (30, 100))
    dir_binary = dir_thresh(img_gray, 15, (0.7, 1.3))
    l_channel, l_binary = hls_channel(img, 'l', (200, 255))
    s_channel, s_binary = hls_channel(img, 's', (90, 255))

    combined_binary = np.zeros_like(img_gray)
    combined_binary[(xsobel_binary == 1) |
                    (l_binary == 1) |
                    (s_binary == 1)] = 255

    # Save the binary image
    cv2.imwrite('output_images/image_binary.jpg', combined_binary)
    # Display images
    f, ax = plt.subplots(5, 2, figsize=(6, 15))
    f.tight_layout()
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original Image', fontsize=12)
    ax[0, 1].imshow(combined_binary, cmap='gray')
    ax[0, 1].set_title('Combine Binary Image', fontsize=12)

    ax[1, 0].imshow(xsobel_binary, cmap='gray')
    ax[1, 0].set_title('X Gradient Binary Image', fontsize=12)
    ax[1, 1].imshow(ysobel_binary, cmap='gray')
    ax[1, 1].set_title('Y Gradient Binary Image', fontsize=12)

    ax[2, 0].imshow(mag_binary, cmap='gray')
    ax[2, 0].set_title('Magnitude Gradient Binary Image', fontsize=12)
    ax[2, 1].imshow(dir_binary, cmap='gray')
    ax[2, 1].set_title('Direction Gradient Binary Image', fontsize=12)

    ax[3, 0].imshow(l_channel, cmap='gray')
    ax[3, 0].set_title('L Channel', fontsize=12)
    ax[3, 1].imshow(l_binary, cmap='gray')
    ax[3, 1].set_title('L Binary Image', fontsize=12)

    ax[4, 0].imshow(s_channel, cmap='gray')
    ax[4, 0].set_title('S Channel', fontsize=12)
    ax[4, 1].imshow(s_binary, cmap='gray')
    ax[4, 1].set_title('S Binary Image', fontsize=12)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


image_undist = mpimg.imread('output_images/test1.jpg')
combinations(image_undist)
