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
				
			
	
transform_data()






def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'r') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte
