RELEASE=23.04
WORKSPACE=/ldap_home/leon.yao/code/work/fastertransformer_backend
#docker pull nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk
docker run --gpus all --rm -it --net host \
	-v ${WORKSPACE}:/workspace \
	nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

