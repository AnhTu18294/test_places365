import numpy as np
import sys
import caffe
import pickle

#path to index and image data files:
fpath_index = '/video/trecvid/sin15/2016t/tshots/keylist1.txt'
fpath_data = '/video/trecvid/sin15/2016t/jpg/'
fpath_outputs = 'outputs/'

# fetch pretrained models
fpath_design = 'models_places/deploy_resnet152_places365.prototxt'
fpath_weights = 'models_places/resnet152_places365.caffemodel'
fpath_labels = 'resources/labels.pkl'

def predictions_scene(fpath_design, fpath_weights, fpath_labels, im):
	# initilaize net
	net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)

	# load input and configure preprocessing
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load('/root/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))  # TODO - remove hardcoded path
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	# since we classify only one image, we change batch size from 10 to 1
	net.blobs['data'].reshape(1,3,224,224)

	# load the image in the data layer
	net.blobs['data'].data[...] = transformer.preprocess('data', im)

	#compute
	out = net.forward()

	# get all of probability accuracy for each image

	return out["prob"].astype(float)





f_out1 = open(fpath_outputs + 'predictions', 'w')
f_out2 = open(fpath_outputs + 'dumpfile_predictions', 'w')

result = {}

with open(fpath_index, 'r') as f_in:
	image_index = f_in.readline()
	while image_index:
		image_index = image_index.replace('\n', '')
		image_file_path = fpath_data + image_index
		im = caffe.io.load_image(image_file_path)

		result[image_index] = predictions_scene(fpath_design, fpath_weights, fpath_labels, im)

		image_index = f_in.readline()

f_out1.write(str(result))
f_out1.close()

pickle.dump(result, f_out2)
f_out2.close()

print 'done!'

# labels = pickle.load(f)
# with open(fpath_outputs + 'labels', 'w') as f_out2:
# 	f_out2.write(labels)