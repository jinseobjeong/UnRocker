#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
#from tensorflow.keras import backend as K

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
#conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
conversion_params = conversion_params._replace(precision_mode="FP16")
#conversion_params = conversion_params._replace(maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='./model2'
    ,conversion_params=conversion_params
    )
converter.convert()
def my_input_fn():
  for _ in range(1):
    inp1 = np.random.uniform(0,1, size=(1, 1024, 1)).astype(np.float32)
#    inp2 = np.random.uniform(0,1, size=(1024, 1)).astype(np.float32)
#    inp1 = np.random.normal(size=(1024, 1)).astype(np.float32)
#    inp2 = np.random.normal(size=(1024, 1)).astype(np.float32)
 
    yield inp1#, inp2
#converter.build(input_fn=my_input_fn)
converter.save(output_saved_model_dir='./model3')

#tf.compat.v1.keras.experimental.export_saved_model(converter, './model3')

#saved_model_loaded = tf.saved_model.load('./model2', tags=[tag_constants.SERVING])
#graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
#output = frozen_func(input_data)[0].numpy()


