# -*- coding: utf-8 -*-
import json
import numpy as np
import triton_python_backend_utils as pb_utils

from pathlib import Path
from typing import List, Union


MAX_BATCH_SIZE = 8


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        self.model_config = model_config = json.loads(args['model_config'])
        if 'parameters' in model_config:
            tokenizer_path = model_config['parameters']['tokenizer_path']['string_value']
            print(f"tokenizer_path: {tokenizer_path}")
        else:
            tokenizer_path = "/workspace/models/llama-7b-hf" 
        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype= pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        from transformers import LlamaTokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        # self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            tokens_batch = pb_utils.get_input_tensor_by_name(request, 'TOKENS_BATCH').as_numpy()
            sequence_length = pb_utils.get_input_tensor_by_name(request, 'sequence_length').as_numpy()
            input_lengths = pb_utils.get_input_tensor_by_name(request, 'input_lengths').as_numpy()

            # Postprocessing output data.
            outputs, new_seq_lengths = self._postprocessing(tokens_batch, sequence_length, input_lengths)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(outputs).astype(self.output_dtype))
            new_seq_lengths_tensor = pb_utils.Tensor(
                'new_seq_lengths',
                new_seq_lengths)
            
            #print(f"outputs: {outputs}")
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                output_tensor,
                new_seq_lengths_tensor,
                ])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


    def _postprocessing(self, tokens_batch, sequence_length, input_lengths):
        cur_folder = Path(__file__).parent
        input_lengths = np.repeat(input_lengths, sequence_length.shape[1], axis=1)
        new_seq_lengths = sequence_length - input_lengths
        outputs = []
        for beam_tokens, beam_len, input_len in zip(tokens_batch, sequence_length, input_lengths):
            for tokens, len, len2 in zip(beam_tokens, beam_len, input_len):
                output = self.tokenizer.decode(tokens[len2 : len], skip_special_tokens=True)
                outputs.append(output.encode('utf8'))
        return outputs, new_seq_lengths