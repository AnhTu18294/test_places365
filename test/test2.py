# input: deploy_resnet152_places365_1.prototxt (after input a data layer in the top of deploy_resnet152_places365 file). index1.txt, python script
# output: Feature from prop layer
# compare to the result that is taken by using extract_feat_dim3_b.cpp


import numpy as np
import sys
import caffe
import pickle
import time

caffe.set_mode_gpu() 

batch_size = 3
num_images = 23
height = 224
width = 224
dim_feature = 3

# fetch pretrained models
fpath_design = '../models_places/deploy_resnet152_places365_1.prototxt'
fpath_weights = '../models_places/resnet152_places365.caffemodel'
fpath_labels = '../resources/labels.pkl'
fpath_output = 'output/test2_python.bin'
num_batchs = num_images/batch_size
rest_images = num_images - num_batchs*batch_size

# initilaize net
net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)
net.blobs['data'].reshape(batch_size,dim_feature,height,width)

fout_prob = open(fpath_output, 'w')

for i in range (0, num_batchs):
	print (i + 1)
	net.forward()["prob"].tofile(fout_prob, '')

if(rest_images != 0):
	print rest_images
	net.forward()["prob"][0: rest_images, :].tofile(fout_prob, '')

print 'done!'
