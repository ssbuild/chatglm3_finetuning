# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]


MODELS_MAP = {
    'chatglm3-6b': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b',
        'config_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b',
    },

    'chatglm3-6b-int4': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-int4',
        'config_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-int4',
    },

    'chatglm3-6b-32k': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-32k',
        'config_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-32k/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-32k',
    },

    'chatglm3-6b-32k-int4': {
        'model_type': 'chatglm',
        'model_name_or_path': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-32k-int4',
        'config_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-32k-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/chatglm3/chatglm3-6b-32k-int4',
    },


}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

