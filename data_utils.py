# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import copy
import json
import os
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser
from data_processer import DataStrategy, TokenIdsMaker
from aigc_zoo.model_zoo.chatglm3.llm_model import ChatGLMTokenizer,PetlArguments,ChatGLMConfig
from config import *

assert train_info_args['max_seq_length'] > 20

data_conf = {
   'strategy': DataStrategy.truncation, # 数据策略选项
    DataStrategy.truncation: {
        'sup': True, # 是否监督训练
    },
    DataStrategy.siding: {
        'sliding_size': train_info_args['max_seq_length'] // 3 * 2, #prompt滑动窗口大小
        'sup': True, # 是否监督训练
        "src_max_length": train_info_args['max_seq_length'] - 10,
        "dst_max_length": None,
    },

}


def preprocess(text):
  #text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  # return text.replace("\\n", "\n").replace("\\t", "\t")
  return text

def build_masks_and_position_ids_glm(batch_input_ids, ctxlens):
    max_len = batch_input_ids.size(1)
    batch_position_ids, batch_attention_mask = [], []
    for input_ids,ctxlen in zip(batch_input_ids,ctxlens):
        position_ids = list(range(0,max_len))
        assert ctxlen <= max_len
        attention_mask = [1] * ctxlen + [0] * (max_len - ctxlen)
        batch_position_ids.append(torch.tensor(position_ids,dtype=torch.long))
        batch_attention_mask.append(torch.tensor(attention_mask,dtype=torch.long))

    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_position_ids = torch.stack(batch_position_ids, dim=0)
    return batch_attention_mask,batch_position_ids

class NN_DataHelper(DataHelper):
    index = 1
    tokens_ids_maker = None
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1


        tokenizer: ChatGLMTokenizer = self.tokenizer # noqa
        config: ChatGLMConfig = self.config           # noqa
        max_seq_length = self.max_seq_length_dict[mode]

        if self.tokens_ids_maker is None:
            self.tokens_ids_maker = TokenIdsMaker(tokenizer=tokenizer,config=config)


        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.truncation:
            ds = self.tokens_ids_maker.trunction(tokenizer,config,examples=examples, max_seq_length=max_seq_length,**data_conf[strategy])
        elif strategy == DataStrategy.siding:
            ds = self.tokens_ids_maker.slidding(tokenizer,config, examples=examples, max_seq_length=max_seq_length, **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy',strategy)

        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    def _get_paragraph(self, lines):
        D = [ ]
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            paragraph = jd[ 'paragraph' ]
            if line_id < 10:
                print(paragraph)


            paragraph = [ (session.get("role",""),session.get("tools",None) ,preprocess(session[ 'q' ]),
                           preprocess('\n'.join(session[ 'a' ])) if isinstance(session[ 'a' ], list) else preprocess(
                               session[ 'a' ]))
                          for session in paragraph ]
            sub = [ ]
            # 自行做模板
            for (role ,tools, q, a) in paragraph:
                # 不是system prompt  answer 必须存在
                if role != "system":
                    assert len(a), ValueError('answer cannot empty')
                sub.append((role,tools,q, a))
            D.append(copy.deepcopy(sub))
            sub.clear()
        return D


    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = [ ]
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            D.extend(self._get_paragraph(lines))
        return D

    def collate_fn(self,batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlens = o.pop('seqlen')
        max_len = torch.max(seqlens).tolist()
        input_ids = o['input_ids'][:, :max_len]

        attention_mask,position_ids = build_masks_and_position_ids_glm(input_ids,seqlens)
        o['input_ids'] = input_ids.long()
        o['attention_mask'] = attention_mask.long()
        o['position_ids'] = position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True,
                                              mode='train',schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval',schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test',schema=schema)

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
        model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)
    elif global_args["trainer_backend"] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args, allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args,allow_extra_keys=True,)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                  config_class_name=ChatGLMConfig)
    



    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
