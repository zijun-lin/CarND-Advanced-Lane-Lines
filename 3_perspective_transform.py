import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# TODO 3. Apply a perspective transform to rectify binary image ("birds-eye view").
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


def perspective_transform_show(img, src, dst):
    # Draw lines in original image
    cv2.line(img, tuple(src[0]), tuple(src[1]), color=(255, 0, 0), thickness=1)
    cv2.line(img, tuple(src[1]), tuple(src[2]), color=(255, 0, 0), thickness=1)
    cv2.line(img, tuple(src[2]), tuple(src[3]), color=(255, 0, 0), thickness=1)
    cv2.line(img, tuple(src[3]), tuple(src[0]), color=(255, 0, 0), thickness=1)
    # Perspective transform
    warped_img = warper(img, src, dst)
    # Display picture
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    f.tight_layout()
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Undistorted Image with source points drawn', fontsize=15)
    ax2.imshow(warped_img, cmap='gray')
    ax2.set_title('Warped result with dest. points drawn', fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def perspective_transform_save(img, src, dst):
    # Perspective transform
    img_warped = warper(img, src, dst)
    cv2.imwrite('output_images/image_warped.jpg', img_warped)


image_show = mpimg.imread('output_images/straight_lines1.jpg')
image_binary = mpimg.imread('output_images/image_binary.jpg')
src_arr = np.float32([[582, 460], [205, 720], [1108, 720], [700, 460]])
dst_arr = np.float32([[320,   0], [320, 720], [960,  720], [960,   0]])
perspective_transform_show(image_show, src_arr, dst_arr)
perspective_transform_save(image_binary, src_arr, dst_arr)
