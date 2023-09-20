output_model_repo="/home/leon.yao/code/fastertransformer_backend_v0.6/perf_analyse/model_analyse_output"
model_repo_path="/home/leon.yao/code/fastertransformer_backend_v0.6/"

mkdir perf_analyse && cd perf_analyse
git init && git remote add -f origin https://github.com/triton-inference-server/model_analyzer.git
git config core.sparseCheckout true && \
echo 'examples' >> .git/info/sparse-checkout && \
git pull origin main

docker pull nvcr.io/nvidia/tritonserver:23.05-py3-sdk
docker run -it --gpus all \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v ${model_repo_path}:/workspace/ \
      -v $(pwd)/examples/quick-start:/workspace/examples/quick-start \
      -v ${output_model_repo}:/workspace/model_analyse_output/ \
      --net=host nvcr.io/nvidia/tritonserver:23.05-py3-sdk
