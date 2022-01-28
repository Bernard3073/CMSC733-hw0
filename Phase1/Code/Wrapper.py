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

	ax = np.linspace(-(size-1) / 2, (size- 1) / 2, size)
	xx, yy = np.meshgrid(ax, ax)
	# kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
	# return kernel / np.sum(kernel)
	kernel = (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp(-(xx**2 + yy**2)/(2*(sigma**2)))
	return kernel

def DoG_filter():
	sigma = [1, 3]
	orient = np.arange(0, 360, 20)
	k_size = 5
	horizontal_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	res = []
	for i in range(len(sigma)):
		DoG = scipy.signal.convolve2d(gauss_kernel(k_size, sigma[i]), horizontal_filter)
		for j in range(len(orient)):
			DoG_rot = scipy.ndimage.rotate(DoG, angle=orient[j], reshape=False)
			res.append(DoG_rot)
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
	gamma = 1
	phi = 1
	lamda = theta
	x = y = np.arange(-k_size, k_size+1)
	x_prime = x * np.cos(theta) + y * np.sin(theta)
	y_prime = x * np.sin(theta) + y * np.cos(theta)

	res = np.exp(-(x_prime**2 + (gamma**2) * (y_prime**2))/(2*(sigma**2))) * np.cos(2*np.pi *(x_prime/lamda) + phi)
	
	return res

def genGabor(sz, omega, theta):
    rad = (int(sz[0]/2), int(sz[1]/2))
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
	for (theta, omega) in params:
		Gabor = genGabor((128,128),omega,theta)
		FilterBank.append(Gabor)
	plt.figure()
	n = len(FilterBank)
	for i in range(n):
		plt.subplot(5,8,i+1)
		plt.axis('off')
		plt.imshow(FilterBank[i],cmap='gray')
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

def chi_square(input, bins, mask_1, mask_2):
	res = np.zeros((input.shape[0], input.shape[1], 12))
	# input_binary = input * 0
	for i in range(12):
		chi_sq = np.zeros((input.shape))
		for j in range(1, bins):
			binary = np.where(input == j, 1, 0)
			g = scipy.signal.convolve2d(binary, mask_1[i], 'same')
			h = scipy.signal.convolve2d(binary, mask_2[i], 'same')
			chi_sq = chi_sq + ((g - h)**2) / (g+h+0.0001)
		res[:, :, i] = chi_sq
	return res

def main():
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	DoG_bank = DoG_filter()
	plt.figure(figsize=(16,2))
	for i in range(2):
		for j in range(16):
			plt.subplot(2, 16, 16*i+j+1)
			plt.axis('off')
			plt.imshow(DoG_bank[16*i+j], cmap='gray')
	plt.savefig('DoG.png')
	plt.show()
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LM_bank = LM_filter()
	plt.figure(figsize=(6, 8))
	for i in range(0, 48):
		plt.subplot(9, 6, i+1)
		plt.axis('off')
		plt.imshow(LM_bank[:, :, i], cmap='gray')
	plt.savefig('LM.png')
	plt.show()

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
	
	Gabor_bank = Gabor_filter_bank()

	filter_bank = []
	count = 0
	for i in range(len(DoG_bank)):
		filter_bank.append(DoG_bank[i])
		count = count+1

	for i in range(LM_bank.shape[2]):
		filter_bank.append(LM_bank[:,:,i])
		count = count+1

	for i in range(len(Gabor_bank)):
		filter_bank.append(Gabor_bank[i])
		count = count+1

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disk_radii = np.array([5, 10, 20])
	orient = np.arange(0, 360, 45)
	mask_1_list = []
	mask_2_list = []
	for i in range(len(half_disk_radii)):
		half_disk = half_disk_mask(half_disk_radii[i])
		for j in range(len(orient)):
			mask_1 = scipy.ndimage.rotate(half_disk, angle=orient[j], reshape=False)
			mask_2 = scipy.ndimage.rotate(mask_1, angle=90, reshape=False)
			mask_1_list.append(mask_1)
			mask_2_list.append(mask_2)
			plt.subplot(6, 8, 16*i+j+1)
			plt.axis('off')
			plt.imshow(mask_1, cmap='gray')
			plt.subplot(6, 8, 16*i+j+1+8)
			plt.axis('off')
			plt.imshow(mask_2, cmap='gray')
	plt.savefig('HDMasks.png')
	plt.show()
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
	file_dir = "../BSDS500/Images/"
	data = [i for i in os.listdir(file_dir) if i.endswith('.jpg')]
	data.sort()
	# img_list = []
	scale_percent = 40
	for i in range(len(data)):
		file_name = file_dir + data[i]
		img = cv2.imread(file_name)
		# img_list.append(img)
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		tex_map = np.zeros((img_gray.size, len(filter_bank)))
		for j in range(len(filter_bank)):
			DoG = cv2.filter2D(img_gray, -1, filter_bank[j])
			DoG = DoG.flatten()
			# DoG = DoG.reshape((1, img_gray.size))
			tex_map[:, j] = DoG
			"""
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		k_means = sklearn.cluster.KMeans(n_clusters=64, n_init=4)
		k_means.fit(tex_map)
		labels = k_means.labels_
		# labels = k_means.predict(tex_map)
		tex_map = np.reshape(labels, (img_gray.shape))
		plt.imshow(tex_map)
		plt.savefig('TextonMap_ImageName.png')
		plt.show()
		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Tg = chi_square(tex_map, 64, mask_1_list, mask_2_list)
		Tg = np.mean(Tg, axis=2)
		plt.imshow(Tg)
		plt.savefig('Tg_IMageName.png')
		plt.show()

		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		br_map = img_gray.reshape((img_gray.size), 1)
		k_means = sklearn.cluster.KMeans(n_clusters=16, random_state=4)
		k_means.fit(br_map)
		labels = k_means.labels_
		br_map = np.reshape(labels, (img_gray.shape[0], img_gray.shape[1]))
		low = np.min(br_map)
		high = np.max(br_map)
		br_map = 255*(br_map - low)/np.float(high - low)

		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Bg = chi_square(br_map, 16, mask_1_list, mask_2_list)
		Bg = np.mean(Bg, axis=2)
		plt.imshow(Bg, cmap='gray')
		plt.show()
		plt.savefig('Bg_ImageName.png')

		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		color_map = img.reshape((img.shape[0]*img.shape[1]),3)
		k_means.fit(color_map)
		labels = k_means.labels_
		color_map = np.reshape(labels, (img.shape[0], img.shape[1]))

		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		Cg = chi_square(color_map, 16, mask_1_list, mask_2_list)
		Cg = np.mean(Cg, axis=2)
		plt.imshow(Cg)
		plt.show()

		"""
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
		file_name = '../BSDS500/SobelBaseline/' + data[i][:-4] + '.png'
		sobel_base = cv2.imread(file_name, 0)
		# img_list.append(img)
		width = int(sobel_base.shape[1] * scale_percent / 100)
		height = int(sobel_base.shape[0] * scale_percent / 100)
		sobel_base = cv2.resize(sobel_base, (width, height), interpolation = cv2.INTER_AREA)
		"""
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
		file_name = '../BSDS500/CannyBaseline/' + data[i][:-4] + '.png'
		canny_base = cv2.imread(file_name, 0)
		# img_list.append(img)
		width = int(canny_base.shape[1] * scale_percent / 100)
		height = int(canny_base.shape[0] * scale_percent / 100)
		canny_base = cv2.resize(canny_base, (width, height), interpolation = cv2.INTER_AREA)
		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		w = 0.5
		pb_lite = ((Tg + Bg + Cg)/3) * (w * canny_base + (1-w) * sobel_base)
		plt.imshow(pb_lite, cmap='gray')
		plt.show()


if __name__ == '__main__':
	main()
