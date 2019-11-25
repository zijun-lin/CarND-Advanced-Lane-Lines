import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def find_lane_pixels(img):
    # Take a histogram of the bottom half of the image
    img_half = img[img.shape[0] // 2:, :]/255
    histogram = np.sum(img_half, axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    left_base = np.argmax(histogram[0: midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])  # y-coordinate of nonzero points
    nonzerox = np.array(nonzero[1])  # x-coordinate of nonzero points

    # Current positions to be updated later for each window in nwindows
    leftx_current = left_base
    rightx_current = right_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        # print('Num: ', window, win_y_high, win_y_low)
        # TODO: Find the four below boundaries of the window
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # TODO: Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzerox >= win_xleft_low) &
                          (nonzerox <= win_xleft_high) &
                          (nonzeroy >= win_y_low) &
                          (nonzeroy <= win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) &
                           (nonzerox <= win_xright_high) &
                           (nonzeroy >= win_y_low) &
                           (nonzeroy <= win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # TODO: If you found > minpix pixels, recenter next window
        #  (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


###################################################################
# TODO: Determine the curvature of the lane and
#  vehicle position with respect to center.
# Load our image
image = mpimg.imread('output_images/image_warped.jpg')

# Find our lane pixels first
leftx, lefty, rightx, righty, out_img = find_lane_pixels(image)

# Fit a second order polynomial to each using `np.polyfit`
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
try:
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
except:
    # Avoids an error if `left` and `right_fit` are still none or incorrect
    print('The function failed to fit a line!')
    left_fitx = 1*ploty**2 + 1*ploty
    right_fitx = 1*ploty**2 + 1*ploty


# Calculate the radius of curvature based on pixel values
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**(3/2)) / (2*np.abs(left_fit[0]))
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**(3/2)) / (2*np.abs(right_fit[0]))
print('Pixel Left Curve: ', left_curverad)
print('Pixel Right Curve: ', right_curverad)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

# Calculate the radius of curvature based on real world
y_eval = np.max(ploty)*ym_per_pix
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**(3/2)) / (2*np.abs(left_fit[0]))
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**(3/2)) / (2*np.abs(right_fit[0]))
print('Real Left Curve: ', left_curverad)
print('Real Right Curve: ', right_curverad)


# Calculate the offset from center
y_pos = image.shape[0]
base_left = left_fit[0] * y_pos * y_pos + left_fit[1] * y_pos + left_fit[2]
base_right = right_fit[0] * y_pos * y_pos + right_fit[1] * y_pos + right_fit[2]
car_pos = image.shape[0]/2
offset = ((base_right - base_left)/2 - car_pos) * xm_per_pix
print('Car Offset: ', offset)


#################################################################
# TODO: Warp the detected lane boundaries back onto the original image.
left_pos = np.array([left_fitx, ploty]).T
right_pos = np.array([right_fitx, ploty]).T
right_pos = np.flipud(right_pos)
pos = np.vstack((left_pos, right_pos))

# new image
color_warp = np.dstack((image, image, image))
cv2.fillPoly(color_warp, np.int_([pos]), (0, 255, 0))

# transf
src = np.float32([[582, 460], [205, 720], [1108, 720], [700, 460]])
dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
newwarp = warper(color_warp, dst, src)

origin_img = mpimg.imread('output_images/test1.jpg')
result = cv2.addWeighted(origin_img, 1, newwarp, 0.3, 0)

# Add text
cv2.putText(result, 'left curvature: ' + str(round(left_curverad, 1)) + ' m',
            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(result, 'right curvature: ' + str(round(right_curverad, 1)) + ' m',
            (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(result, 'offset: ' + str(round(offset * 100., 1)) + ' cm',
            (50, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
f.tight_layout()
ax1.imshow(origin_img)
ax1.set_title('Origin Image', fontsize=15)
ax2.imshow(result)
ax2.set_title('Result Image', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
