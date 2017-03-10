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
fpath_design = 'models_places/deploy_resnet152_places365_1.prototxt'
fpath_weights = 'models_places/resnet152_places365.caffemodel'
fpath_labels = 'resources/labels.pkl'

num_batchs = num_images/batch_size
rest_images = num_images - num_batchs*batch_size

# initilaize net
net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)
net.blobs['data'].reshape(batch_size,dim_feature,height,width)

fout_prob = open('outputs/prob.bin', 'w')

for i in range (0, num_batchs):
	print (i + 1)
	net.forward()["prob"].tofile(fout_prob, '')

if(rest_images != 0):
	print rest_images
	net.blobs['data'].reshape(rest_images,dim_feature,height,width)
	net.forward()["prob"].tofile(fout_prob, '')
	print '-----------'
	print net.forward()["prob"][0: rest_images, :]

print 'done!'
