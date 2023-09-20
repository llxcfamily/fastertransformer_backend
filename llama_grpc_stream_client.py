import argparse

import os
import sys
import csv
import torch
import numpy as np
import json
import multiprocessing as mp
import numpy as np
import time

from functools import partial
from transformers import LlamaForCausalLM, LlamaTokenizer

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


import google.protobuf.json_format
from collections.abc import Mapping
from tritonclient.grpc.service_pb2 import ModelInferResponse


PWD = os.path.abspath(os.path.dirname(__file__))


def _token2inputs(FLAGS, token):
    input_ids = token.input_ids.numpy().astype(np.uint32)
    mem_seq_len = torch.sum(token.attention_mask, dim=1).numpy().astype(np.uint32)
    mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0],1])
    request_output_len = FLAGS.maximum_output_length * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
    runtime_top_k = (FLAGS.sampling_topk * np.ones([input_ids.shape[0],1])).astype(np.uint32)
    runtime_top_p = FLAGS.sampling_topp * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    temperature = 0.85 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    random_seed = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint64)
    is_return_log_probs = True * np.ones([input_ids.shape[0], 1]).astype(bool)
    bad_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
    stop_words_ids = np.array([[[2], [2]]] * input_ids.shape[0], dtype=np.int32)
    beam_width = (FLAGS.beam_width * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
    start_ids = 1 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
    end_ids = 2 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)

    def to_input(name, np_input):
        protocol = "grpc"
        client_util = httpclient if protocol == "http" else grpcclient
        t = client_util.InferInput(
            name, np_input.shape, np_to_triton_dtype(np_input.dtype))
        t.set_data_from_numpy(np_input)
        return t        

    inputs = [to_input("input_ids", input_ids),
              to_input("input_lengths", mem_seq_len),
              to_input("request_output_len", request_output_len),
              to_input("runtime_top_k", runtime_top_k),
              to_input("runtime_top_p", runtime_top_p),
              to_input("beam_search_diversity_rate", beam_search_diversity_rate),
              to_input("temperature", temperature),
              to_input("len_penalty", len_penalty),
              to_input("repetition_penalty", repetition_penalty),
              to_input("random_seed", random_seed),
              to_input("is_return_log_probs", is_return_log_probs),
              to_input("beam_width", beam_width),
              to_input("start_id", start_ids),
              to_input("end_id", end_ids),
              to_input("bad_words_list", bad_words_ids),
              to_input("stop_words_list", stop_words_ids)]
    return inputs


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)
      
def stream_consumer(queue, tokenizer, test: bool):
    start_time = None
    tokens_before = np.array([], dtype=np.int32)
    while True:
        result = queue.get()
        if result is None:
            break

        if isinstance(result, float):
            start_time = result
            continue

        message = ModelInferResponse()
        google.protobuf.json_format.Parse(json.dumps(result), message)
        result = grpcclient.InferResult(message)

        idx = result.as_numpy("sequence_length")[0, 0]
        tokens = result.as_numpy("output_ids")[0, 0, :idx]
        token_text = tokenizer.decode(tokens, skip_special_tokens=True)
        if not token_text.endswith("�"):
            print("[After {:.2f}s] Partial result :\n{}\n".format(
                time.perf_counter() - start_time, token_text))
        
        if test:
            assert len(tokens) == len(tokens_before) + 1
            assert np.array_equal(tokens[:-1], tokens_before)
            tokens_before = tokens

  
def token_printer(tokenizer, result, error):
    if error:
        print("[E:%s]".format(str(result)), end="")      # without "\n"
    else:
        result = grpcclient.InferResult(result.get_response())
        seq_len = result.as_numpy("sequence_length")[0, 0]
        token = np.array([result.as_numpy("output_ids")[0, 0, seq_len-1]])
        token_text = tokenizer.decode(token)
    sys.stdout.flush()
    
def stream_callback(queue, result, error):
    if error:
        queue.put(error)
    else:
        queue.put(result.get_response(as_json=True))


def inference(FLAGS, input_text):
    torch.set_printoptions(precision=6)
    request_parallelism = 1
    verbose = 0
    
    tokenizer = LlamaTokenizer.from_pretrained(FLAGS.hf_model_location, cache_dir="models", padding_side='left')
    
    result_queue = mp.Queue()
    consumer = mp.Process(target=stream_consumer, args=(result_queue, tokenizer, False))
    consumer.start()

    start = time.time() 
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=request_parallelism,
                                        verbose=verbose) as client:
        print("======> input_text:{}".format(input_text))
        inputs = _token2inputs(FLAGS, tokenizer(input_text, return_tensors='pt', padding=True))
        print(f"\033[0;30;41mPrompt\033[0;31m {input_text} ")

        # async stream infer(callback will print it)
        client.start_stream(callback=partial(stream_callback, result_queue))
        result_queue.put(time.perf_counter())
        client.async_stream_infer(FLAGS.model_name, inputs)
    
    result_queue.put(None)
    consumer.join()
    end = time.time()
    print("time cost:{}".format(end - start))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='127.0.0.1:7072',
                        help='Inference server Appkey. Default is .')
    parser.add_argument('-pro', '--protocol', type=str, required=False, default='grpc',
                        help='Inference server Appkey. Default is .')
    parser.add_argument('--hf_model_location', type=str,
                        default="models/llama-7b-hf-converted/tokenizer/",
                        help="tokenizer model path")
    parser.add_argument('-max_output_len', '--maximum_output_length', type=int, default=128, metavar='NUMBER',
                        help='maximum output length (default: 128)')
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='Beam width for beam search. If setting 1, then using sampling.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=5, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.85, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('--model_name', type=str, default="fastertransformer",
                        help='model_name')
    FLAGS = parser.parse_args()

    def generate_prompt(input_text):
        prompt = ''
        prompt +=  "<Human>: "+input_text.strip()+" <AI>: "
        return prompt
    
    # Infer
    input_text = generate_prompt("who is the best nba player?")
    inference(FLAGS, input_text)

