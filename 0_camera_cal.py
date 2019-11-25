import pickle
import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# TODO: 0. Compute the camera calibration matrix and distortion
#  coefficients given a set of chessboard images.
def camera_calibration():
    global img_size
    nx = 9  # TODO: enter the number of inside corners in x
    ny = 6  # TODO: enter the number of inside corners in y
    plt.pause(1)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        img_size = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('Img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    # takes an image, object points, and image points performs the camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size[:2], None, None)

    if ret:  # Save camera calibration parameters
        dist_pickle = {'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
        pickle.dump(dist_pickle, open('mtx_dist_pickle.p', 'wb'))
        print('Save parameters')

    # distortion-corrected image
    for img_name in os.listdir('test_images/'):
        image = cv2.imread('test_images/' + img_name)
        # Undistorting the image:
        img_undist = cv2.undistort(image, mtx, dist, None, mtx)
        cv2.imshow('img_name', img_undist)
        cv2.imwrite('output_images/' + img_name, img_undist)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Display
    image = cv2.imread('camera_cal/calibration1.jpg')
    img_undist = cv2.undistort(image, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(img_undist)
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    print("Undistorted Finish")


camera_calibration()
print('Camera Calibration Finish')
