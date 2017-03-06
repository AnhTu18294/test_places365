import numpy as np
import sys
import caffe
import pickle
import time

#path to index and image data files:
fpath_index = '/home/anhtu/Desktop/places365/docker/index.txt'
fpath_data = '/home/anhtu/Desktop/places365/docker/'
fpath_outputs = '/home/anhtu/Desktop/places365/docker/outputs/'

# fetch pretrained models
fpath_design = '/home/anhtu/Desktop/places365/docker/models_places/deploy_alexnet_places365.prototxt'
fpath_weights = '/home/anhtu/Desktop/places365/docker/models_places/alexnet_places365.caffemodel'
fpath_labels = '/home/anhtu/Desktop/places365/docker/resources/labels.pkl'

# predictions function
def predictions_scene(net, im):	
	# load the image in the data layer
	net.blobs['data'].data[...] = transformer.preprocess('data', im)

	#compute
	out = net.forward()

	# get all of probability accuracy for each image
	return out["prob"].astype(float)

# initilaize net
net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/root/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))  # TODO - remove hardcoded path
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

f_out1 = open(fpath_outputs + 'predictions', 'w')
# f_out3 = open(fpath_outputs + 'dumpfile_predictions', 'w')

with open(fpath_index, 'r') as f_in:
	t1 = time.time()
	image_index = f_in.readline()
	index = 0
	while image_index:
		index += 1
		if((index%2) == 0):
			print 'Image {} in processing .....'.format(index)
		image_index = image_index.replace('\n', '')
		image_file_path = fpath_data + image_index
		im = caffe.io.load_image(image_file_path)
		probs = predictions_scene(net, im)
		f_out1.write(image_index + ':\n' + str(probs) + '\n')
		image_index = f_in.readline()

f_out1.close()

print 'time: {} '.format(time.time() - t1)

# print result

# pickle.dump(result, f_out3)
# f_out1.write(str(result))

# f_out3.close()


# temp = f1.readline()
# while temp:
# 	index = temp.split('/')
# 	print index[0], index[1]
# 	temp = f1.readline()

# labels = pickle.load(f)
# with open(fpath_outputs + 'labels', 'w') as f_out2:
# 	f_out2.write(labels)