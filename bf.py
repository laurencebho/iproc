import numpy as np
import cv2

def main(name, sigma_d, sigma_r, diam, savename):
	#window = "img"
	img = cv2.imread(name, 1)
	if img is not None:
		filtered = cv2.bilateralFilter(img, diam, sigma_r, sigma_d)
		cv2.imwrite(savename, filtered)
		#cv2.imshow(window, filtered)
		#key = cv2.waitKey(0)
		#cv2.destroyAllWindows()
	else:
		print("no image file")

def edge_overlay(original_img, edge_img, savename):
	original = cv2.imread(original_img, 1)
	original = cv2.split(original)
	edges = cv2.imread(edge_img, 0)
	edges = cv2.Canny(edges, 100, 200)
	overlayed = []
	for i in range(3):
		overlayed.append(cv2.add(original[i], edges))
	cv2.imwrite(savename, cv2.merge(overlayed))

if __name__ == "__main__":
	'''
	d = [15, 20, 25]
	r = [30, 50, 80]
	for i in range(3):
		for j in range(3):
			main('./test2.png', d[i], r[j], 9, 'bf2{i}{j}.png'.format(i=i, j=j))
	'''
	main('./test2.png', 25, 10, 9, './edges.png')
	edge_overlay('./test2.png', './edges.png', './overlay2.png')