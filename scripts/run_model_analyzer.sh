
#--triton-http-endpoint=localhost:8070 --triton-grpc-endpoint=localhost:8072 --triton-metrics-url=localhost:8071/metrics \
model-analyzer profile \
    --model-repository /workspace/all_models/llama \
    --profile-models ensemble --triton-launch-mode=docker \
    --output-model-repository-path /workspace/model_analyse_output/ensemble \
    --export-path profile_results \
    --triton-output-path /workspace/model_log.txt \
    --override-output-model-repository \
    --run-config-profile-models-concurrently-enable \
    --run-config-search-mode quick
