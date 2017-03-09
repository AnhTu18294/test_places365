/*----------------------------------------------------------------------------------*/
/*      File: extract_feat_dim3_b.c $                                                */
/*      Author: Bahjat Safadi                                                       */
/*      Description:                                                                */
/*----------------------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;



using namespace caffe; 
/*---------------------------------------------------------------------------------*/
/*                               HEADER FUNCTIONS                                  */
/*---------------------------------------------------------------------------------*/
int count_lines(const char* file_in, size_t *nk);

int generate_feature_extraction_proto(const char *val_templet, const char *list_file, int bachSize, char *ext_proto);
int generate_feature_source_file(const char *path, const char *iFrames, const char * video, size_t nbImages, char *list_file);
template<typename Dtype> int feature_extraction_pipeline(int GPU, int device_id, char* binary_proto, char *ext_proto, char *blob, char *fileOut, size_t nbImages, int bachSize);
/*---------------------------------------------------------------------------------*/
/*                        COMMAND-LINE PARSER FUNCTIONS                            */
/*---------------------------------------------------------------------------------*/
/* find the index of option in the argvs */
static int arg_index(int argc, char *argv[], const char *opt, int n)
{
  int i;
  for (i = 1; i < argc-n; i++) if (!strcmp(opt, argv[i])) return(i);
  return(0);
}
/*------------------------------------------------------------------------------------*/
/* remove string from the input args */
static void arg_remove(int *pargc, char *argv[], int ind, int n)
{
  int i;
  *pargc -= n+1;
  for (i = ind; i < *pargc; i++) argv[i] = argv[i+n+1];
}
/*------------------------------------------------------------------------------------*/
/* read a string ofter the option as input */
char *get_targ(int *pargc, char *argv[], const char *opt, const char *sarg)
{ 
  int i;
  char *s = 0;
  if ((i = arg_index(*pargc,argv,opt,1))) {
    asprintf(&s,"%s",argv[i+1]);
    arg_remove(pargc,argv,i,1);
  } 
  else s = sarg ? strdup(sarg) : NULL;
  
  return(s);
}
/*------------------------------------------------------------------------------------*/
/* read the option as boolean */
int get_barg(int *pargc, char *argv[], const char *opt)
{
	int i;
	if ((i = arg_index(*pargc,argv,opt,0))) arg_remove(pargc,argv,i,0);
	return(i != 0);
}
/*------------------------------------------------------------------------------------*/
/* read a float ofter the option as input */
int get_farg(int *pargc, char *argv[], const char *opt, float fdef, float *pfarg)
{
  int i;
  *pfarg = fdef;
  if ((i = arg_index(*pargc,argv,opt,1))) {
    if (sscanf(argv[i+1],"%f",pfarg) != 1) {
      fprintf(stderr,"ERROR: %s option expects a valid float\n",opt);
      return(0);
    }
    arg_remove(pargc,argv,i,1);
  }

  return(1);
}
/*------------------------------------------------------------------------------------*/
/* read a integer ofter the option as input */
int get_iarg(int *pargc, char *argv[], const char *opt, int fdef, int *pfarg)
{
  int i;
  *pfarg = fdef;
  if ((i = arg_index(*pargc,argv,opt,1))) {
    if (sscanf(argv[i+1],"%d",pfarg) != 1) {
      fprintf(stderr,"ERROR: %s option expects a valid float\n",opt);
      return(0);
    }
    arg_remove(pargc,argv,i,1);
  }

  return(1);
}
/*------------------------------------------------------------------------------------*/
/* read a size_t ofter the option as input */
int get_zuarg(int *pargc, char *argv[], const char *opt, size_t fdef, size_t *pfarg)
{
  int i;
  *pfarg = fdef;
  if ((i = arg_index(*pargc,argv,opt,1))) {
    if (sscanf(argv[i+1],"%zu",pfarg) != 1) {
      fprintf(stderr,"ERROR: %s option expects a valid float\n",opt);
      return(0);
    }
    arg_remove(pargc,argv,i,1);
  }

  return(1);
}
/*--------------------------------------------------------------------------------*/
/*                               TOOL-FUNCTIONS                                   */
/*--------------------------------------------------------------------------------*/
/* counts lines in a text file */
int count_lines(const char* file_in, size_t *nk)
{
  FILE *file;
  *nk = 0; 
  if ((file = fopen(file_in,"r")) == NULL)    return(0);
  while (!feof(file)) *nk += (getc(file) == '\n');
   
  fclose(file);
  return(1);
}
/*-------------------------------------------------------------------------------*/

int generate_feature_extraction_proto(const char *val_templet, const char *list_file, int bachSize, char *ext_proto)
{
  size_t n = 0;
  char *line=0;
  FILE *fp, *fpw;
  /* open files. */
  if (!(fp = fopen(val_templet, "rm")) ||
      !(fpw = fopen(ext_proto, "wm")))    return(0);
  /* Write the new val_file with the list_file as source and the used batch_size. */
  while(!feof(fp) && (getline(&line,&n,fp) > 0)) {
	if (strstr(line,"source:")) fprintf (fpw,"\tsource: \"%s\"\n",list_file);
	else if (strstr(line,"batch_size:")) fprintf (fpw,"\tbatch_size: %d\n",bachSize);
	else fprintf (fpw,"%s",line);
  }

  free(line);
  fclose(fp);
  fclose(fpw);

  return(1);
}

/*-------------------------------------------------------------------------------*/
/* Caffe feature_extraction_pipeline 
*/
int feature_extraction_pipeline(int GPU, int device_id, char* binary_proto, char *ext_proto, char *blob, char *fileOut, size_t nbImages, int bachSize) 
{
  int i,n,d,x,y, numBatches, rest_images, numFeat, dimFeat;
  size_t sizeBlob;
  float *vecOut, *v;
  FILE *fpOut;
  if (GPU) {
	CHECK_GE(device_id, 0);
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } 
  else  Caffe::set_mode(Caffe::CPU);
  
 // Caffe::set_phase(Caffe::TEST);
  
  shared_ptr<Net<float> > feature_extraction_net (new Net<float>(ext_proto,caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(binary_proto);
  
  if(!(fpOut = fopen(fileOut, "w"))) return(0);
  
  CHECK(feature_extraction_net->has_blob(blob))
      << "Unknown feature blob name " << blob
      << " in the network " << ext_proto;

  numBatches = nbImages / bachSize;
  rest_images = nbImages % bachSize;
  if(rest_images) numBatches++;
  vector<Blob<float>*> input_vec;
  shared_ptr<Blob<float> > feature_blob = feature_extraction_net->blob_by_name(blob);
  numFeat = feature_blob->num();
  dimFeat = feature_blob->count() / (numFeat * feature_blob->height()*feature_blob->width());

  sizeBlob = feature_blob->height() *feature_blob->width()*dimFeat*numFeat;
  if(!(vecOut = (float*) malloc(sizeBlob* sizeof(float)))) return (0);
  for (i = 0; i < numBatches; i++) { 
    if (rest_images && (i == numBatches -1)) {
	  numFeat = rest_images;
	  sizeBlob = feature_blob->height() *feature_blob->width()*dimFeat*numFeat;
	}
    feature_extraction_net->Forward(input_vec);
    feature_blob = feature_extraction_net->blob_by_name(blob);

   	for (v = vecOut, n = 0; n < numFeat; n++)
	  for (d = 0; d < dimFeat; d++)
        for (x = 0; x < feature_blob->height(); x++)
          for (y = 0; y < feature_blob->width(); y++, v++) *v = feature_blob->data_at(n,d,x,y);
    if(fwrite (vecOut , sizeof(float), sizeBlob, fpOut) != sizeBlob) return (0);
  }
  free(vecOut);
  fclose(fpOut);
  return 1;
}
/*-------------------------------------------------------------------------------*/
/* Usage. */
void USAGE (int stat)
{    
   printf("USAGE :\n"
	"\t-list string: list file contains the test images with labels.\n"
	"\t-gpu id: in case using a GPU, provide the gpu number\n"
	"\t-bin_proto caffeModel file: the caffe pretrained model for a NN\n"
	"\t-ext_proto ext_proto file: caffe feature extraction prototype file for a NN\n"
	"\t-blob blob Name: the output layer from caffe Net (default fc8)\n"
	"\t-batch batch_size: the batch size for the caffe Net (default 25)\n"
	"\t-out string: the output file (the decriptor name).\n"
	"\tNote: please add the path to the external libraries, like cuda, atlas, mkl..etc. (export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/intel/mkl/lib/intel64/')\n"
	""); 
    exit(stat);
}
/*-------------------------------------------------------------------------------*/
/*                               ... MAIN ...                                    */
/*-------------------------------------------------------------------------------*/
int main(int argc,char* argv[])
{
  char *list_file, *bin_proto, *ext_proto_templet, *ext_proto, *blob, *caffe_outFile; //"BG_14213";//"out";//
  int gpu, batchSize;
  size_t nbImages;

  if(strcmp(argv[1],"--h")==0 || strcmp(argv[1],"-h")==0) { 
	USAGE (0);
  }
  /* Parse command-line. */
  if (!(list_file = get_targ(&argc,argv,"-list",NULL)))   
  {
	fprintf(stderr, "error while parsing command-line, please provide -list listfile\n"); 
	USAGE(1);
  }
  if(!(bin_proto = get_targ(&argc,argv,"-bin_proto", NULL))) {
	fprintf(stderr, "error while parsing command-line, please provide -bin_proto\n"); 
	USAGE(1);
  }
  if(!(ext_proto_templet = get_targ(&argc,argv,"-ext_proto", NULL))) {
	fprintf(stderr, "error while parsing command-line, please provide -ext_proto\n"); 
	USAGE(1);
  }
  if (!(caffe_outFile = get_targ(&argc,argv,"-out",NULL)))   
  {
	fprintf(stderr, "error while parsing command-line, please provide -out\n"); 
	USAGE(1);
  }
  
  blob = get_targ(&argc,argv,"-blob", "fc8");
  if (!(get_iarg(&argc,argv,"-gpu",-1,&gpu))||
	  !(get_iarg(&argc,argv,"-batch",25,&batchSize))) {
	fprintf(stderr, "error while parsing command-line, please provide correct values for -gpu and -batch\n"); 
	USAGE(1);
  }
 
  if(!(count_lines(list_file, &nbImages)))
  {
	fprintf(stderr, "error while reading list_file :%s.\n",list_file); 
	exit(1);
  }
  /* Generate file paths. */
  ext_proto = 0;
  asprintf(&ext_proto,"%s.ext_proto",list_file);
  /* Generate the feature_extraction_prototype file for Caffe. */
  printf("Processing with caffe concepts.\n"); fflush(stdout);
  if(!generate_feature_extraction_proto(ext_proto_templet, list_file, batchSize, ext_proto)) exit(1);
  /* Use caffe tool to extract imageNet features for each iframe in the video. */
  if(!(feature_extraction_pipeline ((gpu >= 0) , gpu, bin_proto, ext_proto, blob, caffe_outFile, nbImages, batchSize)))
  {
	fprintf(stderr, "error. The system failed to extract features using caffe tool.\n"); 
	exit(1);
  }
  /*Remove temporal files and folders. */
  remove(ext_proto);
  /* Free memory. */
  free(bin_proto);
  free(blob);
  free(ext_proto);
  free(ext_proto_templet);
  free(caffe_outFile);
  /*Exit.*/
  printf("Done.\n");

  exit(0);
}