import numpy as np
import sys
import caffe
import pickle
import time

caffe.set_mode_gpu() 
#path to index and image data files:
fpath_index = 'index1.txt'
fpath_data = ''
fpath_outputs = 'outputs0/'

# fetch pretrained models
fpath_design = 'models_places/deploy_resnet152_places365.prototxt'
fpath_weights = 'models_places/resnet152_places365.caffemodel'
fpath_labels = 'resources/labels.pkl'

# predictions function
def predictions_scene(net, im):
	# load the image in the data layer
	net.blobs['data'].data[...] = transformer.preprocess('data', im)

	#compute
	net.forward()

	# get all of probability accuracy for each image
	return net.blobs["fc365"].data[0]

# initilaize net
net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/home/mrim/nguyenan/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))  # TODO - remove hardcoded path
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,224,224)

f_out = open(fpath_outputs + 'predictions', 'w')

with open(fpath_index, 'r') as f_in:
	t1 = time.time()
	image_index = f_in.readline().replace(' 0', '')
	index = 0
	while image_index:
		index += 1
		if((index%1000) == 0):
			print 'Image {} in processing .....'.format(index)
		image_index = image_index.replace('\n', '')
		image_file_path = fpath_data + image_index
		im = caffe.io.load_image(image_file_path)
		probs = predictions_scene(net, im)
		probs.tofile(f_out, '')
		# f_out.write(image_index + ':\n'+ str(probs))
		image_index = f_in.readline()

f_out.close()

print 'done!'
