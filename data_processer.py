# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import json
import random
import typing
from enum import Enum
import numpy as np
from aigc_zoo.model_zoo.chatglm3.llm_model import ChatGLMTokenizer

class DataStrategy(Enum):
    truncation = 1
    siding = 2

class TokenIdsMaker:
    def __init__(self,tokenizer: ChatGLMTokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.bos_token_id = self.tokenizer.get_command("<bos>")
        self.pad_token_id = self.tokenizer.get_command("<pad>")
        self.eos_token_id = self.tokenizer.get_command("<eos>")

    def build_single_message(self, role, metadata, message):
        assert role in ["system", "user", "assistant", "observation"], role
        role_tokens = [self.tokenizer.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        tokens = role_tokens + message_tokens
        return tokens

    @classmethod
    def final(cls, input_ids: typing.List, labels, max_seq_length, tokenizer):
        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen

        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))

        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen,
        }
        return d


    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
        input_ids = []
        for item in history:
            content = item["content"]
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
            input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        if query is not None:
            input_ids.extend(self.build_single_message(role, "", query))
        return self.tokenizer.encode(input_ids,  is_split_into_words=True)

    def parse_history_from_answers(self, output, history):
        content = ""
        metadata = ""
        history = copy.deepcopy(history)
        for response in output.split("<|assistant|>"):
            #ensure 语料包含换行符 格式为{metadata}\n{content}
            if response.find('\n') == -1:
                response = '\n' + response
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
        return metadata, content, history

    def trunction(self, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length,sup=True):
        ds = []
        history = []
        for sid, (q_role,q,a) in enumerate(examples):
            if q_role == "system":
                history += [{
                    "role": "system",
                    "content": q,
                }]
                continue

            history += [ {
                "role": "user",
                "content": q,
            } ]
            metadata, content, history = self.parse_history_from_answers(a,history)
            a_ids = self.build_chat_input(query=None,history=history)
            b_ids = self.tokenizer.encode(content)
            role_tokens = [ self.tokenizer.get_command("<|assistant|>") ] + self.tokenizer.encode(f"{metadata}\n")
            while len(a_ids) + len(b_ids) > max_seq_length - len(role_tokens) - 2:
                if len(b_ids) > len(a_ids):
                    b_ids.pop(-1)
                else:
                    a_ids.pop(0)
            assert len(b_ids) > 0
            b_ids += [ self.eos_token_id ]
            a_ids = a_ids
            # ensure answer format
            b_ids = role_tokens + b_ids
            input_ids = a_ids + b_ids
            labels = copy.deepcopy(input_ids) if not sup else [ -100 ] * len(a_ids) + copy.deepcopy(b_ids)
            input_ids = [self.bos_token_id] + input_ids
            labels = [self.bos_token_id] + labels if not sup else [ -100 ]  + labels
            assert len(input_ids) <= max_seq_length
            ds.append(self.final(input_ids, labels, max_seq_length, tokenizer))

        return ds



    # def slidding(cls, tokenizer: ChatGLMTokenizer,config, messages,
    #              max_seq_length,
    #              sliding_size = None,
    #              src_max_length=-1,
    #              dst_max_length=-1,
    #              sup=True):
    #
    #
    #     if sliding_size is None or sliding_size < 0:
    #         sliding_size = max_seq_length - 1
    #
    #     assert sliding_size <= max_seq_length - 1
    #
    #     ds = []
    #
    #     for sid, (q, a) in enumerate(messages):
    #         a_ids = tokenizer.encode(text=build_template(q,prefix=prefix, history=examples[:sid]), add_special_tokens=False)
    #         b_ids = tokenizer.encode(text=a, add_special_tokens=False)
    #         if src_max_length and src_max_length > 0:
    #             a_ids = a_ids[:src_max_length]
    #         if dst_max_length and dst_max_length > 0:
    #             b_ids = b_ids[:dst_max_length]
    #
    #         b_ids += [config.eos_token_id]
    #         input_ids_qa = a_ids + b_ids
    #         labels_all = copy.deepcopy(input_ids_qa) if not sup else [-100] * len(a_ids) + b_ids
    #
    #         pos = 0
    #         while pos < len(input_ids_qa):
    #             input_ids = input_ids_qa[pos:pos + max_seq_length - len(sptoken)]
    #             labels = labels_all[pos:pos + max_seq_length - len(sptoken)]
    #
    #             pos += sliding_size
    #             if np.all(np.asarray(labels) == -100):
    #                 continue
    #
    #             input_ids = sptoken + input_ids
    #             labels = sptoken + labels if not sup else [-100] * len(sptoken) + labels
    #             ds.append(cls.final(input_ids,labels,max_seq_length,tokenizer))
    #     return ds
