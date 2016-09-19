import numpy as np
import cv2
from ROF import solve_ROF

def add_noise(img) :
	#add gaussian noise
	img = img + 0.2 * np.random.normal(size = img.shape)
	return img
	

if(__name__ == '__main__') :
	img_ref = cv2.imread('Lenna.jpg', 0)
	img_ref = img_ref / 255.0
	img_obs = add_noise(img_ref)
	img_ROF = solve_ROF(img_obs, 5.0)
	
	cv2.imwrite('origin.png', img_ref * 255.0)
	cv2.imwrite('observe.png', img_obs * 255.0)
	cv2.imwrite('ROF.png', img_ROF * 255.0)
	#cv2.imshow('ori', img_ref)
	#cv2.imshow('obs', img_obs)
	#cv2.imshow('ROF', img_ROF)
	#cv2.waitKey(0)