import numpy as np
import math
import cv2

def main(noisy, flash, sigma_d, sigma_r, rad, savename):
	A = cv2.imread("./" + noisy, cv2.IMREAD_COLOR)
	F = cv2.imread("./" + flash, cv2.IMREAD_COLOR)
	A_channels = cv2.split(A)
	F_channels = cv2.split(F)
	height = A.shape[0]
	width = A.shape[1]
	img = cv2.split(A)
	for i in range(3):
		print(i)
		img[i]  =joint_bilateral(A_channels[i], F_channels[i], img[i], rad, sigma_d, sigma_r, height, width)
	img = cv2.merge(img)
	cv2.imwrite(savename, img)

def joint_bilateral(A, F, img, rad, sigma_d, sigma_r, height, width):
	diam = rad * 2 + 1
	height = A.shape[0]
	width = A.shape[1]
	mask = create_mask(rad, sigma_d)
	for y in range(height - 1):
		for x in range(width - 1):
			img[y][x] = filter(y, x, A, F, mask, rad, diam, sigma_r, height, width)
	return img

def create_mask(rad, sigma):
	diam = rad * 2 + 1
	mask = np.zeros(shape=(diam, diam))
	for i in range(-rad, rad  + 1):
		for j in range(-rad, rad  + 1):
			mask[i+rad, j+rad] = gaussian((i ** 2 + j ** 2) ** 0.5, sigma)
	sum  = np.sum(mask)
	mask = np.divide(mask, sum)
	return mask

def get_neighbours(img, y, x, rad, height, width):
	diam = rad * 2 + 1
	top = rad if y - rad >= 0 else y
	bottom = rad + 1 if y + rad < height else height - y
	left = rad if x - rad >= 0 else x
	right = rad + 1 if x + rad < width else width - x
	neighbours = np.zeros(shape=(diam, diam))
	neighbours[rad - top:rad + bottom, rad - left:rad + right] = img[y - top:y + bottom, x - left:x + right]
	return neighbours

def filter(y, x, A, F, mask, rad, diam, sigma_r, height, width):
	F_neighbours = get_neighbours(F, y, x, rad, height, width)
	pix = F_neighbours[rad][rad]
	A_neighbours = get_neighbours(A, y, x, rad, height, width)
	result = np.copy(mask)
	result = np.multiply(result, gaussian_mask(np.subtract(F_neighbours, pix) , sigma_r))
	k = np.sum(result)
	result = np.multiply(result, A_neighbours)
	result = np.divide(result, k)
	return np.sum(result)

def gaussian_mask(mask, sigma):
	divisor = -2 * sigma ** 2
	mask = np.square(mask)
	mask = np.divide(mask, divisor)
	mask = np.exp(mask)
	return mask

def gaussian(x, sigma):
	return math.exp((-1 * x **2) / (2 * sigma ** 2))# / (sigma * (2 * math.pi) ** 0.5)

if __name__ == '__main__':
	main("test3a.jpg", "test3b.jpg", 5, 1, 20, "./test.jpg")