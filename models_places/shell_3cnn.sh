if [ "$1" = '-h' -o "$1" = '--h' ] 
then
	echo "USAGE: "
	echo "-1 list images"
	echo "-2  gpu index"
	echo "-3 output folder"
	echo "-4 IMG in case and the list is a path to an image"
	echo "example in case of a list file: ./shell_3cnn.sh  list_img.txt 0  test/LIG/"
	echo "example in case of an image file: ./shell_3cnn.sh  image.jpg 0  test/LIG/ IMG"
else
	export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/usr/local/lib:/usr/lib64/:/opt/intel/lib/intel64/:/home/mrim/dutta/caffe-cudnn/build/lib/
	#Programs
	prog_extract_feat="/home/mrim/safadi/caffe-dvorak/caffe_cpu/extract_feat_dim3_b.bin"
	efuse0="/home/guest/groups/irim/quenot/bin/efuse0"
	prog_pca="/home/guest/groups/irim/quenot/bin/pcarot"
	prog_powvec2="/home/guest/groups/irim/quenot/bin/pwsvec" #"/data1/home/guest/groups/irim/quenot/bin/powvec2"
	msvm_predict="/home/mrim/safadi/svn/quaero/prog/src_libs/libsvm-plus-3.20/msvm-predict"
	#Global paths
	caffe_model="/home/mrim/safadi/caffe-dvorak/caffe-models/"
	pca_gpath="/video/trecvid/sin15/2013/tshots/LIG/"
	msvm_model="/home/mrim/safadi/SIN2013/MSVM_2015/LIG_alex_goog_vgg_earlypw1.000p294pw0.610.model13d"
	svi="/video/trecvid/sin15/2013/tshots/LIG/alex_goog_vgg_earlypw1.000p294pw0.610.bin"
	
	#The three descriptors and their parameters (dim, alpha1,2, k-pca,...)
	bin=("$caffe_model/alex/caffe_reference_imagenet_model" "$caffe_model/bvlc_googlenet/bvlc_googlenet.caffemodel" "$caffe_model/vgg/VGG_ILSVRC_19_layers.caffemodel")
	ext=("$caffe_model/alex/alex_val.prototxt" "$caffe_model/bvlc_googlenet/val.prototxt" "$caffe_model/vgg/val_if25.prototxt")
	out=("caffe_fc6n_4096" "googlenet_pool5b_1024" "vgg_all_fc8")
	blob=("fc6" "pool5/7x7_s1" "fc8")
	dim=(4096 1024 1000)
	a1=("0.449" "0.650" "0.950")
	a2=("0.655" "0.670" "0.400")
	pca1=(662 660 1000)
	pca2=(662 660 609)
	batch=(50 50 25)
	#input values
	gpu=$2
	outpath=$3
	img=0;
	
	if [ ! -d $outpath ]
	then
		mkdir -p $outpath
	fi

	if [ "$4" = 'IMG' ]
	then
		img=1;
	fi
	if [ "$img" = "1" ]
	then
		list=$outpath"/tmpList.txt"
		echo $1" 0" >$list
	else
		list=$1
	fi	
	nbImas=`wc -l $list|awk '{print $1}'`
	
	N=3
	for((i=0; i<N; i++)); do
		$prog_extract_feat \
		-list $list -gpu $gpu -batch ${batch[i]} \
		-bin_proto ${bin[i]} \
		-ext_proto ${ext[i]} \
		-blob ${blob[i]} \
		-out ${outpath}${out[i]}.bin
		
		$prog_pca -n $nbImas -d ${dim[i]} \
		-plaw ${a1[i]} \
		-fi ${outpath}${out[i]}.bin \
		-favg ${pca_gpath}${out[i]}pw${a1[i]}.avg \
		-fevec ${pca_gpath}${out[i]}pw${a1[i]}p${pca1[i]}.evec \
		-feval ${pca_gpath}${out[i]}pw${a1[i]}p${pca1[i]}.eval -dout ${pca2[i]} -fo ${outpath}${out[i]}pw${a1[i]}p${pca2[i]}.bin
		
		$prog_powvec2 -id ${pca2[i]} -fi ${outpath}${out[i]}pw${a1[i]}p${pca2[i]}.bin -fs ${pca_gpath}${out[i]}pw${a1[i]}p${pca2[i]}.scale  \
		              -od ${pca2[i]} -fo ${outpath}${out[i]}pw${a1[i]}p${pca2[i]}pw${a2[i]}.bin -plaw ${a2[i]}
		rm ${outpath}${out[i]}pw${a1[i]}p${pca2[i]}.bin 
	done
	
	if [ -f $outpath/alex_goog_vgg_early.e ]
	then
		rm $outpath/alex_goog_vgg_early.e
	fi
	for((i=0; i<N; i++)); do
		echo ${outpath}${out[i]}pw${a1[i]}p${pca2[i]}pw${a2[i]}.bin >>$outpath/alex_goog_vgg_early.e 
	done
	
	
	$efuse0 $list $outpath/alex_goog_vgg_early.e $outpath/alex_goog_vgg_early.bin
	rm $outpath/alex_goog_vgg_early.e
	if [ "$img" = "1" ]
	then
		rm $list
	fi
	$prog_pca -n $nbImas -d 1931 -plaw 1.000 \
	-fi ${outpath}/alex_goog_vgg_early.bin \
	-favg ${pca_gpath}/alex_goog_vgg_earlypw1.000.avg \
	-fevec ${pca_gpath}/alex_goog_vgg_earlypw1.000p1931.evec \
	-feval ${pca_gpath}/alex_goog_vgg_earlypw1.000p1931.eval \
	-dout 294 -fo ${outpath}/alex_goog_vgg_earlypw1.000p294.bin	
		
	$prog_powvec2 -id 294 -fi ${outpath}/alex_goog_vgg_earlypw1.000p294.bin -fs ${pca_gpath}/alex_goog_vgg_earlypw1.000p294.scale  \
		          -od 294 -fo ${outpath}/alex_goog_vgg_earlypw1.000p294pw0.610.bin -plaw 0.610
	rm ${outpath}alex_goog_vgg_earlypw1.000p294.bin  
	#predict msvm
	echo "MSVM_PREDICT"
	export  LD_LIBRARY_PATH=/opt/intel/lib/intel64/
	$msvm_predict \
	-OBIN -D -BIN -q -b 1 -v 4096 -DIM 294 \
	-SVI $svi  \
	${outpath}/alex_goog_vgg_earlypw1.000p294pw0.610.bin \
	$msvm_model \
	$outpath/alex_goog_vgg_earlypw1.000p294pw0.610_tv13.bin
fi
#echo "/home/safadi/src_MSVM13/prog/ligsvm_13_predict \
#-dd /video/trecvid/sin15/2013/tshots/LIG/alex_goog_vgg_earlypw1.000p294pw0.610.bin \
#-ka /video/trecvid/sin15/2013/tshots/keylist.txt -kd /video/trecvid/sin15/2013d/tshots/keylist.txt \
#-dt LIG/alex_goog_vgg_earlypw1.000p294pw0.610.bin -kat  ~/Hackday/keylist.txt -kt ~/Hackday/keylist.txt  \
#-sc 0 -cat  /video/trecvid/sin15/all/cat/2013f.txt -kpca 1 \
#-model /home/safadi/SIN2013/MSVM_2015/2013z/LIG/alex_goog_vgg_earlypw1.000p294pw0.610_SVM_t2/fpos_4_s1000_step1_f0_noscale_2.83054_l0/models/ \
#-PROB"
