# input: deploy_resnet152_places365.prototxt. 1 image, python script
# output: Feature from prop layer that is saved at test/output/test1_python.bin

import numpy as np
import caffe

caffe.set_mode_gpu()

fpath_design = '../models_places/deploy_resnet152_places365.prototxt'
fpath_weights = '../models_places/resnet152_places365.caffemodel'
fpath_labels = '../resources/labels.pkl'

fpath_image = '../images/1433.jpg'
fpath_output = 'output/test1_python.bin'

# initilaize net
net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('models_places/places365_mean.npy').mean(1).mean(1))  # TODO - remove hardcoded path
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

im = caffe.io.load_image(fpath_image)
fout = open(fpath_output, 'w')

net.blobs['data'].reshape(1,3,224,224)

net.blobs['data'].data[...] = transformer.preprocess('data', im)

net.forward()["prob"].tofile(fout, '')
fout.close()