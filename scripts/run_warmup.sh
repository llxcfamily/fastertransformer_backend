 echo "model_warmup [{" >> config.pbtxt && \
        echo "    name : \"regular sample\"" >> config.pbtxt && \
        echo "    batch_size: 1" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${INPUT_PREFIX}0\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_FP32" >> config.pbtxt && \
        echo "            dims: 16" >> config.pbtxt && \
        echo "            zero_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "    inputs {" >> config.pbtxt && \
        echo "        key: \"${INPUT_PREFIX}1\"" >> config.pbtxt && \
        echo "        value: {" >> config.pbtxt && \
        echo "            data_type: TYPE_FP32" >> config.pbtxt && \
        echo "            dims: 16" >> config.pbtxt && \
        echo "            random_data: true" >> config.pbtxt && \
        echo "        }" >> config.pbtxt && \
        echo "    }" >> config.pbtxt && \
        echo "}]" >> config.pbtxt


  echo "model_warmup [{" >> config.pbtxt && \
            echo "    name : \"string statefull\"" >> config.pbtxt && \
            echo "    batch_size: 8" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${SEQ_INPUT}\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_STRING" >> config.pbtxt && \
            echo "            dims: 1" >> config.pbtxt && \
            echo "            input_data_file: \"raw_string_data\"" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${START}\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_INT32" >> config.pbtxt && \
            echo "            dims: 1" >> config.pbtxt && \
            echo "            zero_data: true" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "    inputs {" >> config.pbtxt && \
            echo "        key: \"${READY}\"" >> config.pbtxt && \
            echo "        value: {" >> config.pbtxt && \
            echo "            data_type: TYPE_INT32" >> config.pbtxt && \
            echo "            dims: 1" >> config.pbtxt && \
            echo "            zero_data: true" >> config.pbtxt && \
            echo "        }" >> config.pbtxt && \
            echo "    }" >> config.pbtxt && \
            echo "}]" >> config.pbtxt
