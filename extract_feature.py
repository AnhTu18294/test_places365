import numpy as np
import sys, getopt
import optparse
import caffe
from caffe import layers as L

# usage
def USAGE(): 
	print """USAGE :
		\t-h more help
		\t-l string: list file contains the test images with labels.
		\t-H number: new height of image after forward throughs data layer
		\t-W number: new width of image after forward throughs data layer
		\t-p number: crop size of image for preprocessing
		\t-i True/False: rotate image at data layer
		\t-m mean_file: the binary mean file for data image preprocessing
		\t-g number (>= 0): in case using a GPU, provide the gpu number
		\t-c caffeModel file: the caffe pretrained model for a NN
		\t-e ext_proto file: caffe feature extraction prototype file for a NN
		\t-b blobs Name: the list of output layer from caffe Net
		\t-s batch_size: the batch size for the caffe Net (default 25)
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

def get_new_height(new_height):
	try:
		temp = int(new_height)
	except ValueError:
		print "new height must be a number"
		sys.exit()
	else:
		return temp

def get_new_width(new_width):
	try:
		temp = int(new_width)
	except ValueError:
		print "new width must be a number"
		sys.exit()
	else:
		return temp

def get_crop_size(crop_size):
	try:
		temp = int(crop_size)
	except ValueError:
		print "crop size must be a number"
		sys.exit()
	else:
		return temp

def get_mirror(mirror):
	if mirror == 'True':
		return True
	elif mirror == 'False':
		return False
	else:
		print 'ERROR: Mirror must be True or False'
		sys.exit()

def get_batch_size(batch_size):
	try:
		temp = int(batch_size)
	except ValueError:
		print "batch size must be a number"
		sys.exit()
	else:
		return temp


def refactor_then_load_network(input):

	n = caffe.NetSpec()

	if(input['gpu_id'] >= 0):
		n.data, n.lable = L.ImageData(type = 'ImageData', ntop = 2, source = input['source'], mean_file = input['mean_file'], batch_size = input['batch_size'], crop_size = input['crop_size'], mirror = input['mirror'], new_height = input['new_height'], new_width = input['new_width'])
	else:
		n.data, n.lable = L.ImageData(type = 'ImageData', ntop = 2, source = input['source'], mean_file = input['mean_file'], crop_size = input['crop_size'], mirror = input['mirror'], new_height = input['new_height'], new_width = input['new_width'])
	
	data_layer =  str(n.to_proto())

	# open deploy file to refactor
	try:
		f_deploy = open(input['deploy_file_name'], 'r')
	except:
		print "ERROR: Cannot open the file named ", input['deploy_file_name']
		sys.exit()

	# open new val_net file
	try:
		f_val = open('val_net.prototxt', 'w')
	except:
		print "ERROR: Cannot open the buffer val_net file for refactoring network. Please try it again"
		sys.exit()
	
	f_val.write(f_deploy.readline())
	f_val.write(data_layer)
	f_deploy.readline()
	f_deploy.readline()
	f_deploy.readline()
	f_deploy.readline()
	f_deploy.readline()
	line = f_deploy.readline()
	while line:
		f_val.write(line)
		line = f_deploy.readline()

	f_val.close()

	# check and set caffe mode: gpu or cpu	
	if (input['gpu_id'] >= 0):
		# set gpu mode
		caffe.set_mode_gpu()
		caffe.set_device(input['gpu_id'])
	else:
		# set cpu mode
		caffe.set_mode_cpu()

	net = caffe.Net('val_net.prototxt', input['caffe_model'], caffe.TEST)

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
parser.add_option('-H', '--new_height', help='new height of image after forward throughs data layer', default=224)
parser.add_option('-W', '--new_width', help='new width of image after forward throughs data layer', default=224)
parser.add_option('-p', '--crop_size', help='crop size of image for preprocessing', default=224)
parser.add_option('-i', '--mirror', help='rotate image at data layer', default= False)
parser.add_option('-m', '--mean_file', help='the binary mean file for data image preprocessing')
parser.add_option('-g', '--gpu_id', help='in case using a GPU, provide the gpu number. If not, run with cpu', default=-1)
parser.add_option('-c', '--caffe_model', help='caffeModel file: the caffe pretrained model for a NN')
parser.add_option('-e', '--ext_proto', help='ext_proto file: caffe feature extraction prototype file for a NN')
parser.add_option('-b', '--blobs', help='the list of output layer from caffe Net')
parser.add_option('-s', '--batch_size', help='the batch size for the caffe Net', default=25)
parser.add_option('-o', '--out_files', help='the list output file that corresponse with the list output layer')



def main():
	opts, args = parser.parse_args()

	source = opts.list_images
	new_width = get_new_width(opts.new_width)
	new_height = get_new_height(opts.new_height)
	crop_size = get_crop_size(opts.crop_size)
	mirror = get_mirror(opts.mirror)
	mean_file = opts.mean_file
	gpu_id = get_gpu(opts.gpu_id)
	caffe_model = opts.caffe_model
	deploy_file_name = opts.ext_proto
	list_blobs_name = np.array(opts.blobs.replace('[', '').replace(']', '').split(','))
	batch_size = get_batch_size(opts.batch_size)
	list_out = np.array(opts.out_files.replace('[', '').replace(']', '').split(','))
	
	dict_input = {'source': source, 'new_width': new_width, 'new_height': new_height, 'crop_size': crop_size, 'mirror': mirror, 'mean_file': mean_file, 'gpu_id': gpu_id, 'caffe_model': caffe_model, 'deploy_file_name': deploy_file_name, 'batch_size': batch_size}

	# check and count the number of images in source file
	num_images = count_line(source)

	# check and map output file name to list blobs name 
	
	if(list_blobs_name.size != list_out.size):
		print 'ERROR: list output file must have the same size with list blobs'
		sys.exit()
	else:
		ouput_to_blobs = dict(zip(list_out, list_blobs_name))

	net = refactor_then_load_network(dict_input)
	
	blobs_net = dict(net.blobs).keys()
	
	if(is_valided_blobs_name(list_blobs_name, blobs_net) == False):
		sys.exit()


	# check list ouput file name and try to create output file
	f_outs = {}
	try:
		for out in list_out:
			print out
			f_outs[out] = open(out, 'w')
	except:
		print "ERROR: Cannot create the output file named ", out
		sys.exit()

	print ouput_to_blobs
	# extract feature 
	if(gpu_id >= 0):
		print "using cpu to forward image on network"
		num_batchs = num_images/batch_size
		rest_images = num_images - num_batchs*batch_size

		net.blobs['data'].reshape(batch_size,3,new_height, new_width)

		for i in range (0, num_batchs):
			for key in f_outs.keys():
				dict(net.blobs)[ouput_to_blobs[key]].data.tofile(f_outs[key], '')

		if(rest_images != 0):
			for key in f_outs.keys():
				dict(net.blobs)[ouput_to_blobs[key]].data.tofile(f_outs[key], '')
	else:
		print "using cpu to forward image on network"
		net.blobs['data'].reshape(1,3,new_height, new_width)
		for i in range(0, num_images):
			for key in f_outs.keys():
				dict(net.blobs)[ouput_to_blobs[key]].data.tofile(f_outs[key], '')

if __name__ == "__main__":
	if len(sys.argv) == 1:
		USAGE()
		sys.exit()
	main()

