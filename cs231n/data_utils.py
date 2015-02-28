import cPickle as pickle
import gzip
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
from scipy import misc
#rom PIL import Image
from scipy import ndimage

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


def save_horz_flip(img_arr, new_imgpath):
	h_flip = np.fliplr(img_arr)
	try: 
		misc.imsave(new_imgpath, convert_3_channels(h_flip))
		return 1
	except IOError:
		return 0
	
def save_vert_flip(img_arr, new_imgpath):
	v_flip = np.flipud(img_arr)
	try:
		misc.imsave(new_imgpath, convert_3_channels(v_flip))
		return 1
	except IOError:
		return 0
	
def save_rand_rotate(img_arr, new_imgpath):
	rot_angle = np.random.rand(1) * 330 + 15 # between 15 and 345 degrees rotation
	rotate_img = ndimage.rotate(img_arr, rot_angle, reshape=False)
	try:
		misc.imsave(new_imgpath, convert_3_channels(rotate_img))
		return 1
	except IOError:
		return 0
	
#prob defines probability of f2
def write_with_prob(f1, f2, prob, data):
	if np.random.rand(1) < prob:
		f2.write(data)
		return 0
	else:
		f1.write(data)
		return 1




#write training data paths and labels to file for caffe to turn into leveldb 
def outputDataTxtForCaffe():
			
	val_percentile_split = 0.033
	target_sz = 128	
	processed_dirpath_root = 'VGG_augmented_train'
	processed_dirpath = processed_dirpath_root + '/'
	
	suffix = 2
	example_count = 0
	with open('augmented_dataTxtFileForCaffe_train.txt', 'w+') as f_train:
		with open('augmented_dataTxtFileForCaffe_val.txt', 'w+') as f_val:
			for index, class_dir in enumerate(class_dirs):
				class_dirpath = join(dirpath, class_dir)
				for img in listdir(class_dirpath):

					img_path = join(class_dirpath, img)
					if img_path.endswith('.jpg'):
						img_arr = misc.imread(img_path)
						
						#pad and resize
						img_arr = transform_img(img_arr)
						img_arr = misc.imresize(img_arr, (target_sz, target_sz))
						
						new_imgpath = join(processed_dirpath, img)
						
						#write original example
						example_count += 1
						try:
							misc.imsave(new_imgpath, convert_3_channels(img_arr))
							status = write_with_prob(f_train, f_val, val_percentile_split, new_imgpath + ' ' + str(index) + '\n')
						except IOError:
							processed_dirpath = processed_dirpath_root + str(suffix) + '/'
							if not os.path.exists(processed_dirpath_root + str(suffix)):
								os.makedirs(processed_dirpath_root + str(suffix))
							suffix += 1
							new_imgpath = join(processed_dirpath, img)
							misc.imsave(new_imgpath, convert_3_channels(img_arr))
							status = write_with_prob(f_train, f_val, val_percentile_split, new_imgpath + ' ' + str(index) + '\n')
						
						#status 0 means data was used for validation 
						
						if status == 0: continue
						#randomly rotate image
						rotate_prob = np.random.rand(1)
						if rotate_prob > 0.5:
							rot_imgpath = join(processed_dirpath, 'rot_' + img)
							status = save_rand_rotate(img_arr, rot_imgpath)
							if status == 1:
								example_count += 1
								f_train.write(rot_imgpath + ' ' + str(index) + '\n')
						
						#randomly flip horizontally and vertically 
						horz_flip_prob = np.random.rand(1)
						if horz_flip_prob > 0.5:
							horz_imgpath = join(processed_dirpath, 'horz_' + img)
							status = save_horz_flip(img_arr, horz_imgpath)
							if status == 1:
								example_count += 1
								f_train.write(horz_imgpath + ' ' + str(index) + '\n')
						
						vert_flip_prob = np.random.rand(1)
						if vert_flip_prob > 0.5:
							vert_imgpath = join(processed_dirpath, 'vert_' + img)
							status = save_vert_flip(img_arr, vert_imgpath)
							if status == 1:
								example_count += 1
								f_train.write(vert_imgpath + ' ' + str(index) + '\n')

	print "TOTAL EXAMPLES: ", example_count

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
						
		
outputDataTxtForCaffe()
#splitProcessedData()

#Turn on and off to do automatically
#transform_data()