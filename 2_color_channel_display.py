import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def rgb_channel(img_origin, channel='r', thresh=(0, 255)):
    rgb = np.copy(img_origin)
    if 'r' == channel:
        img_channel = rgb[:, :, 0]
    elif 'g' == channel:
        img_channel = rgb[:, :, 1]
    elif 'b' == channel:
        img_channel = rgb[:, :, 1]
    else:
        print('RGB, illegal image channel!')
        return
    # Threshold color channel
    img_binary = np.zeros_like(img_channel)
    img_binary[(img_channel > thresh[0]) & (img_channel <= thresh[1])] = 1
    return img_channel, img_binary


def hls_channel(img_origin, channel='h', thresh=(0, 255)):
    hls = cv2.cvtColor(img_origin, cv2.COLOR_RGB2HLS)
    if 'h' == channel:
        img_channel = hls[:, :, 0]
    elif 'l' == channel:
        img_channel = hls[:, :, 1]
    elif 's' == channel:
        img_channel = hls[:, :, 2]
    else:
        print('HLS, illegal image channel!')
        return
    # Threshold color channel
    img_binary = np.zeros_like(img_channel)
    img_binary[(img_channel > thresh[0]) & (img_channel <= thresh[1])] = 1
    return img_channel, img_binary


def lab_channel(img_origin, channel='l', thresh=(0, 255)):
    lab = cv2.cvtColor(img_origin, cv2.COLOR_RGB2LAB)
    if 'l' == channel:
        img_channel = lab[:, :, 0]
    elif 'a' == channel:
        img_channel = lab[:, :, 1]
    elif 'b' == channel:
        img_channel = lab[:, :, 2]
    else:
        print('LAB, illegal image channel!')
        return
    # Threshold color channel
    img_binary = np.zeros_like(img_channel)
    img_binary[(img_channel > thresh[0]) & (img_channel <= thresh[1])] = 1
    return img_channel, img_binary


def rgb_display(img):
    r_c, r_b = rgb_channel(img, 'r', (230, 255))
    g_c, g_b = rgb_channel(img, 'g', (200, 255))
    b_c, b_b = rgb_channel(img, 'b', (200, 255))
    # Display images
    f, ax = plt.subplots(3, 2, figsize=(8, 16))
    f.tight_layout()
    ax[0, 0].imshow(r_c, cmap='gray')
    ax[0, 0].set_title('R Channel', fontsize=16)
    ax[0, 1].imshow(r_b, cmap='gray')
    ax[0, 1].set_title('R binary', fontsize=16)
    ax[1, 0].imshow(g_c, cmap='gray')
    ax[1, 0].set_title('G Channel', fontsize=16)
    ax[1, 1].imshow(g_b, cmap='gray')
    ax[1, 1].set_title('G Binary', fontsize=16)
    ax[2, 0].imshow(b_c, cmap='gray')
    ax[2, 0].set_title('B Channel', fontsize=16)
    ax[2, 1].imshow(b_b, cmap='gray')
    ax[2, 1].set_title('B Binary', fontsize=16)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def hls_display(img):
    h_c, h_b = hls_channel(img, 'h', (50, 255))
    l_c, l_b = hls_channel(img, 'l', (200, 255))
    s_c, s_b = hls_channel(img, 's', (90, 255))
    # Display images
    f, ax = plt.subplots(3, 2, figsize=(8, 16))
    f.tight_layout()
    ax[0, 0].imshow(h_c, cmap='gray')
    ax[0, 0].set_title('H Channel', fontsize=16)
    ax[0, 1].imshow(h_b, cmap='gray')
    ax[0, 1].set_title('H binary', fontsize=16)
    ax[1, 0].imshow(l_c, cmap='gray')
    ax[1, 0].set_title('L Channel', fontsize=16)
    ax[1, 1].imshow(l_b, cmap='gray')
    ax[1, 1].set_title('L Binary', fontsize=16)
    ax[2, 0].imshow(s_c, cmap='gray')
    ax[2, 0].set_title('S Channel', fontsize=16)
    ax[2, 1].imshow(s_b, cmap='gray')
    ax[2, 1].set_title('S Binary', fontsize=16)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def lab_display(img):
    l_c, l_b = lab_channel(img, 'l', (230, 255))
    a_c, a_b = lab_channel(img, 'a', (200, 255))
    b_c, b_b = lab_channel(img, 'b', (170, 255))
    # Display images
    f, ax = plt.subplots(3, 2, figsize=(8, 16))
    f.tight_layout()
    ax[0, 0].imshow(l_c, cmap='gray')
    ax[0, 0].set_title('L Channel', fontsize=16)
    ax[0, 1].imshow(l_b, cmap='gray')
    ax[0, 1].set_title('L binary', fontsize=16)
    ax[1, 0].imshow(a_c, cmap='gray')
    ax[1, 0].set_title('A Channel', fontsize=16)
    ax[1, 1].imshow(a_b, cmap='gray')
    ax[1, 1].set_title('A Binary', fontsize=16)
    ax[2, 0].imshow(b_c, cmap='gray')
    ax[2, 0].set_title('B Channel', fontsize=16)
    ax[2, 1].imshow(b_b, cmap='gray')
    ax[2, 1].set_title('B Binary', fontsize=16)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


image = mpimg.imread('output_images/test1.jpg')
# rgb_display(image)
hls_display(image)
# lab_display(image)
