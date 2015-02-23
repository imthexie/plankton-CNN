import cPickle as pickle
import gzip
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
from scipy import misc
from PIL import Image

dirpath = "train/"	
class_dirs = [dir for dir in listdir(dirpath) if isdir(join(dirpath,dir))]

def getObjFromPklz(infilename):
    f = gzip.open(infilename, 'rb')
    try:
        return pickle.load(f)
    finally:
        f.close()

def writeToPklz(outfilename, obj):
    output = gzip.open(outfilename, 'wb')
    try:
        pickle.dump(obj, output, -1)
    finally:
        output.close()
		
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
	

def transform_data():
	#build a huge numpy array
	num_classes = len(class_dirs) 
	num_examples = 30336 #empirical finding
	print 'Num Classes:' , num_classes
	print 'Num Examples:', num_examples
	target_sz = 64
	X = np.empty((num_examples, 1, target_sz, target_sz))
	y = np.empty((num_examples, ))
	count = 0
	for index, class_dir in enumerate(class_dirs):
		class_dirpath = join(dirpath, class_dir)
		for img in listdir(class_dirpath):
			img_path = join(class_dirpath, img)
			if img_path.endswith('.jpg'):
				img_arr = misc.imread(img_path)
				img_arr = transform_img(img_arr)
				img_arr = misc.imresize(img_arr, (target_sz, target_sz))
				X[count, 0] = img_arr
				y[count] = index #index is class label
				count +=1
	
	datadict = {'data' : X, 'labels' : y}
	writeToPklz('train_datadict_initial', datadict)


def load_train_data(filename='train_datadict_initial'):
	datadict = getObjFromPklz(filename)
	X = datadict['data']
	Y = datadict['labels']
	return X,Y

#write training data paths and labels to file for caffe to turn into leveldb 
def outputDataTxtForCaffe():
			
	target_sz = 128	
	processed_dirpath = 'processed_train/'
	
	example_count = 0
	with open('dataTxtFileForCaffe_noval_train.txt', 'r+') as f:
		for index, class_dir in enumerate(class_dirs):
			class_dirpath = join(dirpath, class_dir)
			for img in listdir(class_dirpath):
				img_path = join(class_dirpath, img)
				if img_path.endswith('.jpg'):
					img_arr = misc.imread(img_path)
					img_arr = transform_img(img_arr)
					img_arr = misc.imresize(img_arr, (target_sz, target_sz))
					new_imgpath = join(processed_dirpath, img)
					misc.imsave(new_imgpath, img_arr)
					f.write(new_imgpath + ' ' + str(index) + '\n')


def splitProcessedData():
	#split randomly validation set
	num_examples = 30336
	num_validation = 1000
	num_training = num_examples - num_validation
	mask = range(0, num_examples)
	val_mask = np.random.choice(mask, size=num_validation, replace = False)
	val_mask.astype(int)

	with open('dataTxtFileForCaffe.txt', 'r') as f:
		with open('dataTxtFileForCaffe_noval_train.txt', 'w+') as f_train:
			with open('dataTxtFileForCaffe_validation_train.txt', 'w+') as f_val:
				for m in xrange(num_training + num_validation):
					if m in val_mask:
						f_val.write(f.readline())
					else:
						f_train.write(f.readline())
						
		
#outputDataTxtForCaffe()
splitProcessedData()

#Turn on and off to do automatically
#transform_data()