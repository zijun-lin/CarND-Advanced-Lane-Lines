import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

# a = np.array([0.7, 0.5, 0.15])).all()

relative_delta = np.array([0.6, 0.4, 0.1])
a = (abs(relative_delta) < np.array([0.7, 0.5, 0.15])).all()
print(abs(relative_delta))
print(a)