# Make sure that caffe is on the python path:
#caffe_root = '~/caffe/'  # this file is expected to be in {caffe_root}/examples
#import sys, os
#sys.path.insert(0, caffe_root + 'python')

import numpy as np
import caffe
from baseline import get_train_data

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val = get_train_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'caffemodel.prototxt'

#net = caffe.Net(MODEL_FILE)
#caffe.set_mode_gpu()

solver = caffe.SGDSolver('~/plankton-CNN/cs231n/solver.prototxt')
solver.net.set_mode_gpu()
solver.net.set_input_arrays(X_train, y_train)
solver.net.set_input_arrays(X_val, y_val)
solver.solve()