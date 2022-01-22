#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import scipy.ndimage
import scipy.signal
import math
import matplotlib.pyplot as plt
import sklearn.cluster
import os

def gauss_kernel(size, sigma):

	ax = np.linspace(-(size-1) / 2., (size- 1) / 2., size)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

	return kernel / np.sum(kernel)


def DoG_filter():
	sigma = [1, 3]
	orient = np.arange(0, 360, 20)
	k_size = 5
	horizontal_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	res = []
	# plt.figure(figsize=(16,2))
	for i in range(len(sigma)):
		DoG = scipy.signal.convolve2d(gauss_kernel(k_size, sigma[i]), horizontal_filter)
		for j in range(len(orient)):
			DoG_rot = scipy.ndimage.rotate(DoG, angle=orient[j], reshape=False)
			res.append(DoG_rot)
	# 		plt.subplot(len(sigma), len(orient), len(orient)*i+j+1)
	# 		plt.imshow(res[len(orient)*i+j], cmap='gray')
	# plt.show()
	return res


def gauss2d(n, sigma):
	size = int((n-1)/2)
	var = sigma**2
	m = np.asarray([[x**2 + y**2 for x in range(-size, size+1)]
				   for y in range(-size, size+1)])
	output = (1/np.sqrt(2*np.pi*var))*np.exp(-m/(2*var))
	return output


def gauss1d(sigma, mean, x, order):
	x = np.array(x) - mean
	var = sigma**2
	g = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x*x)/(2*var)))

	if order == 0:
		return g
	elif order == 1:
		g = -g*((x)/(var))
		return g
	else:
		g = g*(((x*x) - var)/(var**2))
		return g


def log2d(n, sigma):
	size = int((n-1)/2)
	var = sigma**2
	m = np.asarray([[x**2 + y**2 for x in range(-size, size+1)]
				   for y in range(-size, size+1)])
	n = (1/np.sqrt(2*np.pi*var))*np.exp(-m/(2*var))
	output = n*(m - var)/(var**2)
	return output


def binary(img, bin_value):
	binary_img = img * 0
	for r in range(0, img.shape[0]):
		for c in range(0, img.shape[1]):
			if img[r, c] == bin_value:
				binary_img[r, c] = 1
			else:
				binary_img[r, c] = 0
	return binary_img


def gradient(maps, numbins, mask_l, mask_r):
	gradient = np.zeros((maps.shape[0], maps.shape[1], 12))
	for m in range(0, 12):
		chi = np.zeros((maps.shape))
		for i in range(1, numbins):
			tmp = binary(maps, i)
			g = scipy.signal.convolve2d(tmp, mask_l[m], 'same')
			h = scipy.signal.convolve2d(tmp, mask_r[m], 'same')
			chi = chi + ((g-h)**2) / (g+h+0.0001)
		gradient[:, :, m] = chi
	return gradient


def makefilter(scale, phasex, phasey, pts, sup):
	gx = gauss1d(3*scale, 0, pts[0, ...], phasex)
	gy = gauss1d(scale, 0, pts[1, ...], phasey)
	image = gx*gy
	image = np.reshape(image, (sup, sup))
	return image


def LM_filter():
	sup = 49
	scalex = np.sqrt(2) * np.array([1, 2, 3])
	norient = 6
	nrotinv = 12

	nbar = len(scalex)*norient
	nedge = len(scalex)*norient
	nf = nbar+nedge+nrotinv
	F = np.zeros([sup, sup, nf])
	hsup = (sup - 1)/2

	x = [np.arange(-hsup, hsup+1)]
	y = [np.arange(-hsup, hsup+1)]

	[x, y] = np.meshgrid(x, y)

	orgpts = [x.flatten(), y.flatten()]
	orgpts = np.array(orgpts)

	count = 0
	for scale in range(len(scalex)):
		for orient in range(norient):
			angle = (np.pi * orient)/norient
			c = np.cos(angle)
			s = np.sin(angle)
			rotpts = [[c+0, -s+0], [s+0, c+0]]
			rotpts = np.array(rotpts)
			rotpts = np.dot(rotpts, orgpts)
			F[:, :, count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
			F[:, :, count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
			count = count + 1

	count = nbar+nedge
	scales = np.sqrt(2) * np.array([1, 2, 3, 4])

	for i in range(len(scales)):
		F[:, :, count] = gauss2d(sup, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:, :, count] = log2d(sup, scales[i])
		count = count + 1

	for i in range(len(scales)):
		F[:, :, count] = log2d(sup, 3*scales[i])
		count = count + 1

	return F


def gabor_filter(k_size, theta):
	'''
	a convolution filter representing a combination of gaussian and a sinusoidal term
	'''
	sigma = 500
	gamma = 0.1
	phi = 0.8
	lamda = theta
	x = y = np.arange(-k_size, k_size+1)
	x_prime = x * np.cos(theta) + y * np.sin(theta)
	y_prime = x * np.sin(theta) + y * np.cos(theta)

	res = np.exp(-(x_prime**2 + (gamma**2) * (y_prime**2))/(2*(sigma**2))) * np.cos(2*np.pi *(x_prime/lamda) + phi)
	
	return res

def genGabor(sz, omega, theta):
    rad = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-rad[0], rad[0]+1), range(-rad[1], rad[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega**2 / (4*np.pi * np.pi**2) * np.exp(- omega**2 / (8*np.pi**2) * ( 4 * x1**2 + y1**2))

    sinusoid = np.cos(omega * x1) * np.exp(np.pi**2 / 2)

    gabor = gauss * sinusoid
    return gabor

#Gabor Filter Bank
def Gabor_filter_bank():
	theta = np.arange(1.5*np.pi, 0.5*np.pi, -np.pi/8) 
	omega = np.arange(0.35, 0.1, -0.05)
	params = [(t,o) for o in omega for t in theta]
	FilterBank = []
	gaborParams = []
	for (theta, omega) in params:
		gaborParam = {'omega':omega, 'theta':theta, 'sz':(128, 128)}
		Gabor = genGabor((128,128),omega,theta)
		FilterBank.append(Gabor)
		gaborParams.append(gaborParam)

	plt.figure()
	n = len(FilterBank)
	for i in range(n):
		plt.subplot(5,8,i+1)
		plt.axis('off'); plt.imshow(FilterBank[i],cmap='gray')
	plt.show()
	return FilterBank

def half_disk_mask(rad):
	# res = np.zeros((2*rad, 2*rad))
	# for i in range():
	# 	for j in range():
	# a, b = rad/2, rad/2
	# n = 7
	# r = rad

	# y,x = np.ogrid[-a:n-a, -b:n-b]
	# mask = x*x + y*y <= r*r

	# res = np.zeros((n, n))
	# res[mask] = 1
	hd = np.zeros((rad*2,rad*2))
	rs = rad**2
	for i in range(rad):
		m = (i-rad)**2
		for j in range(2*rad):
			if m+(j-rad)**2 < rs:
				hd[i,j] =1
	return hd

def main():
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	
	DoG_bank = DoG_filter()
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	scales_s = np.array([1, math.sqrt(2), 2, 2*math.sqrt(2)])
	img_lm = LM_filter()
	# plt.figure(figsize=(6, 8))
	# for i in range(0, 48):
	# 	plt.subplot(9, 6, i+1)
	# 	plt.axis('off')
	# 	plt.imshow(img_lm[:, :, i], cmap='gray')
	# plt.show()

	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	# theta = np.arange(0, np.pi, np.pi/4)
	# # omega = np.arange(0.2, 0.6, 0.1)
	# k_size = 5
	# sigma =5
	# gabor_filter_bank = []
	# for t in theta:
	# 	g_filter = gabor_filter(k_size, t)
	# 	gabor_filter_bank.append(g_filter)

	# plt.figure()
	# for i in range(len(gabor_filter_bank)):
	# 	# plt.subplot(4, 4, i+1)
	# 	plt.axis('off')
	# 	plt.imshow(gabor_filter_bank[i], cmap='gray')
	# plt.show()
	
	# Gabor_filters = Gabor_filter_bank()
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disk_radii = np.array([5, 10, 20])
	orient = np.arange(0, 360, 45)
	mask_top_list = []
	mask_bottom_list = []
	for i in range(len(half_disk_radii)):
		half_disk = half_disk_mask(half_disk_radii[i])
		for j in range(len(orient)):
			mask_top = scipy.ndimage.rotate(half_disk, angle=orient[j], reshape=False)
			mask_bottom = scipy.ndimage.rotate(mask_top, angle=90, reshape=False)
			mask_top_list.append(mask_top)
			mask_bottom_list.append(mask_bottom)
	# 		plt.subplot(6, 8, 16*i+j+1)
	# 		plt.axis('off')
	# 		plt.imshow(mask_top, cmap='gray')
	# 		plt.subplot(6, 8, 16*i+j+1+8)
	# 		plt.axis('off')
	# 		plt.imshow(mask_bottom, cmap='gray')
	# plt.show()
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	file_dir = "../BSDS500/Images/"
	img_list = [i for i in os.listdir(file_dir) if i.endswith('.jpg')]
	tex_map = []
	img_list.sort()
	for i in range(len(img_list)):
		for j in range(len(DoG_bank)):
			file_name = file_dir + img_list[i]
			img = cv2.imread(file_name)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			DoG = cv2.filter2D(img_gray, -1, DoG_bank[j])
			tex_map.append(DoG)
	tex_map = np.array(tex_map)
	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	k_means = sklearn.cluster.KMeans(n_clusters=64, n_init=4)
	# k_means.fit(tex_map)
	# labels = k_means.labels_
	# tex_map = np.reshape(labels, (tex_map.shape))
	# plt.imshow(tex_map)
	# plt.show()
	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	"""
	Generate Brightness Map
	Perform brightness binning 
	"""

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	"""
	Generate Color Map
	Perform color binning or clustering
	"""

	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""

	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""

	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
