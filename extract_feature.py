import numpy as np
import sys, getopt, os
import optparse
import caffe
from caffe import layers as L

source = None
new_height = None
new_width = None
batch_size = None
crop_size = None
mean_file = None
caffe_model = None
deploy_file_name = None
list_blobs_name = None
list_out = None

# usage
def USAGE(): 
	print """USAGE :
		\t-h more help
		\t-l string: list file contains the test images with labels.
		\t-m mean_file: the binary mean file for data image preprocessing
		\t-g number (>= 0): in case using a GPU, provide the gpu number
		\t-c caffeModel file: the caffe pretrained model for a NN
		\t-e ext_proto file: caffe feature extraction prototype file for a NN
		\t-b blobs Name: the list of output layer from caffe Net
		\t-s batch_size: the batch size for the caffe Net 
		\t-o string: the output file (the decriptor name).
		\tNote: please add the path to the external libraries, like cuda, atlas, mkl..etc. (export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/intel/mkl/lib/intel64/')\n""" 
	return 

# check caffe mode
def is_gpu_mode():
	global gpu_id
	if (gpu_id >= 0):
		return True
	else:
		return False

def get_gpu(gpu):
	try:
		temp = int(gpu)
	except ValueError:
		print "gpu must be a number"
		sys.exit()
	else:
		return temp

def get_batch_size(batch_size):
	if batch_size is None:
		return None
	try:
		temp = int(batch_size)
	except ValueError:
		print "batch size must be a number"
		sys.exit()
	else:
		return temp


def refactor_then_load_network():
	global source, new_width, new_height, mean_file, gpu_id, caffe_model, deploy_file_name, list_blobs_name, batch_size, list_out

	# open deploy file to refactor
	try:
		f_deploy = open(deploy_file_name, 'r')
	except:
		print "ERROR: Cannot open the file named ", deploy_file_name
		sys.exit()

	network_name = ''
	input_dim = []
	other_attribute = {}
	line = f_deploy.readline()
	while line:
		if('layer {' in line):
			break
		elif ('name:' in line):
			network_name = line[len('name:'): len(line)]
		elif ('input_dim:' in line):
			input_dim.append(int(line[len('input_dim:'):len(line)]))
		else:
			i = 0
			for c in line:
				i += 1
				if c == ':':
					break
			key = line[0:i]
			value = line[i:len(line)]
			other_attribute[key] = value
		line = f_deploy.readline()

	if(len(input_dim) != 4):
		print 'ERROR: input_dim must have 4 values in deploy file'
		sys.exit()

	n = caffe.NetSpec()
	if batch_size is None:
		batch_size = input_dim[0]	
	new_height = input_dim[2]
	new_width = input_dim[3]

	n.data, n.lable = L.ImageData(type = 'ImageData', ntop = 2, source = source, mean_file = mean_file, batch_size = batch_size, crop_size = max(new_width, new_height), mirror = False, new_height = new_height, new_width = new_width)
	data_layer =  str(n.to_proto())

	# open new val_net file
	try:
		f_val = open('val_net.prototxt', 'w')
	except:
		print "ERROR: Cannot open the buffer val_net file for refactoring network. Please try it again"
		sys.exit()

	f_val.write('name:' + network_name +'\n')
	f_val.write(data_layer)
	while line:
		f_val.write(line)
		line = f_deploy.readline()

	f_val.close()
	f_deploy.close()
	
	# check and set caffe mode: gpu or cpu	
	if (gpu_id >= 0):
		caffe.set_mode_gpu()
		caffe.set_device(gpu_id)
	else:
		caffe.set_mode_cpu()

	net = caffe.Net('val_net.prototxt', caffe_model, caffe.TEST)

	# os.remove('val_net.prototxt')

	return net


def is_valided_blobs_name(list_blobs_name, blobs_net):
	res = True
	for name in list_blobs_name:
		if name not in blobs_net:
			print 'ERROR: Cannot find the layer named by ', name
			res = False
	return res

def count_line(file_name):
	res = 0
	try:
		fin = open(file_name, 'r')
	except:
		print "ERROR: cannot find or open the file named ", file_name
		sys.exit()
	else:
		while fin.readline():
			res += 1
	return res

parser = optparse.OptionParser()
parser.add_option('-l', '--list_images',help='file contains the images with labels.')
parser.add_option('-m', '--mean_file', help='the binary mean file for data image preprocessing')
parser.add_option('-g', '--gpu_id', help='in case using a GPU, provide the gpu number. If not, run with cpu', default=-1)
parser.add_option('-c', '--caffe_model', help='caffeModel file: the caffe pretrained model for a NN')
parser.add_option('-e', '--ext_proto', help='ext_proto file: caffe feature extraction prototype file for a NN')
parser.add_option('-b', '--blobs', help='the list of output layer from caffe Net')
parser.add_option('-s', '--batch_size', help='the batch size for the caffe Net')
parser.add_option('-o', '--out_files', help='the list output file that corresponse with the list output layer')


def main():
	global source, mean_file, gpu_id, caffe_model, deploy_file_name, list_blobs_name, batch_size, list_out
	opts, args = parser.parse_args()	
	source = opts.list_images
	mean_file = opts.mean_file
	gpu_id = get_gpu(opts.gpu_id)
	caffe_model = opts.caffe_model
	deploy_file_name = opts.ext_proto
	list_blobs_name = np.array(opts.blobs.replace('[', '').replace(']', '').split(','))
	batch_size = get_batch_size(opts.batch_size)
	list_out = np.array(opts.out_files.replace('[', '').replace(']', '').split(','))
	
	# check and count the number of images in source file
	num_images = count_line(source)

	# check output filename to list blobs name 
	if(list_blobs_name.size != list_out.size):
		print 'ERROR: list output file must have the same size with list blobs'
		sys.exit()

	net = refactor_then_load_network()
	
	blobs_net = dict(net.blobs).keys()
	
	if(is_valided_blobs_name(list_blobs_name, blobs_net) == False):
		sys.exit()
	
	# check list ouput file name and try to create output file
	f_outs = []
	try:
		for out in list_out:
			f_temp = open(out, 'w')
			f_outs.append(f_temp)
	except:
		print "ERROR: Cannot create the output file named ", out
		sys.exit()

	# extract feature 
	num_batchs = num_images/batch_size
	rest_images = num_images - num_batchs*batch_size

	net.blobs['data'].reshape(batch_size,3,new_height, new_width)

	for i in range (0, num_batchs):
		net.forward()
		for i in range(0, len(list_out)):
			net.blobs[list_blobs_name[i]].data.tofile(f_outs[i], '')

	if(rest_images != 0):
		net.forward()
		for i in range(0, len(list_out)):
			net.blobs[list_blobs_name[i]].data[0: rest_images, :].tofile(f_outs[i], '')
	print 'done!'

if __name__ == "__main__":
	if len(sys.argv) == 1:
		USAGE()
		sys.exit()
	main()

