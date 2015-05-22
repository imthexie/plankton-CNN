#assumed that you are in the caffe folder

import os, sys, random
import numpy as np
from scipy.misc import imread, imresize
from os import listdir
from os.path import isfile, join, isdir
import caffe 

caffe.set_device(0)

#This is specifically for VGG data 
def subtract_mean_pixel(img_arr):
	mean_pix = [103.939, 116.779, 123.68];
	for c in xrange(0, 3):
		img_arr[c, :, :] -= mean_pix[c]
	return img_arr 

#convert to 3 channels for VGG
def convert_3_channels(img_arr):
	new_img_arr = np.tile(img_arr, (3, 1, 1))
	return subtract_mean_pixel(new_img_arr)


def transform_img(img_arr):
	h, w = img_arr.shape
	if w > h:
		img_arr = img_arr.T
	elif w == h:
		return img_arr
	
	h, w = img_arr.shape
	
	#we assume h > w
	diff = h - w
	pad1_dim = diff / 2
	pad2_dim = diff - pad1_dim
	if pad1_dim > 0:
		pad1 = 255 * np.ones((h, pad1_dim), dtype=np.uint8)
		img_arr = np.hstack((pad1, img_arr))	
	if pad2_dim > 0:
		pad2 = 255 * np.ones((h, pad2_dim), dtype=np.uint8)
		img_arr = np.hstack((img_arr, pad2))
	assert img_arr.shape[0] == img_arr.shape[1]
	return img_arr

def format_test_data():
	dirpath = "plankton_data/test/"
	processed_dirpath = "plankton_data/test/" #REPLACE
	
	class_dirs = [dir for dir in listdir(dirpath) if isdir(join(dirpath,dir))]

	target_sz = 224	
	with open('test_lmdb_text.txt', 'w+') as f:
		for index, class_dir in enumerate(class_dirs):
			class_dirpath = join(dirpath, class_dir)
			for img in listdir(class_dirpath):

				img_path = join(class_dirpath, img)
				if img_path.endswith('.jpg'):
					img_arr = misc.imread(img_path)
					#pad and resize
					img_arr = transform_img(img_arr)
					img_arr = misc.imresize(img_arr, (target_sz, target_sz))

					#rewrite original example
					misc.imsave(img_path, convert_3_channels(img_arr))
					f.write(img_path + ' ' + str(index) + '\n')
				
def createSubmission():
	#CREATE THE SUBMISSION CSV
	model_file = 'plankton_data/VGG_16.prototxt'
	weights_file1 = 'plankton_data/VGG_sxie_run4_iter_22000.caffemodel'
	net1 = caffe.Net(model_file, weights_file1, caffe.TEST)

	weights_file2 = 'plankton_data/VGG_sxie_run4_iter_14000.caffemodel'
	net2 = caffe.Net(model_file, weights_file2, caffe.TEST)

	weights_file3 = 'plankton_data/VGG_sxie_run4_iter_7000.caffemodel'
	net3 = caffe.Net(model_file, weights_file3, caffe.TEST)
				
	for layer, name in zip(net.layers, net._layer_names):
		  print name, layer.type
		  for blob in layer.blobs:
			print '  ', blob.data.shape
				
			
			
			
			
format_test_data() 
			