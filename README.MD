### 天妒英才！人民的好总理，一路走好 !
```text
    10-29 train and test pass
    10-28 initial chatglm3
```
   
##  statement
- [deep_training](https://github.com/ssbuild/deep_training)

## install
  - pip install -U -r requirements.txt
  - 如果无法安装 , 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt  

```text
dev 通过一下方式安装
pip install -U git+https://github.com/ssbuild/deep_training.git
pip install -U transformers>=4.30 deepspeed xformers bitsandbytes>=0.39 accelerate>=0.20
```


## weight

- [chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)   支持四种微调方式
- [chatglm3-6b-int4](https://huggingface.co/THUDM/chatglm3-6b-int4)  支持四种微调方式
- [chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k)   支持四种微调方式
- [chatglm3-6b-32k-int4](https://huggingface.co/ssbuild/chatglm3-6b-32k-int4/tree/main)  支持四种微调方式



## data sample
- open_data 不定时开放新数据 https://github.com/ssbuild/open_data
- [prompt](assets/PROMPT.md)
- [tools prompt](assets/TOOLS.md)

```text
支持两种数据格式示例如下
例子依次分别是 对话，工具，代码执行，对话，对话

数组组成 
role: 可选字段(str) 标志 q字段 角色, one of user system, observation ，system 标识是否为 system prompt , system prompt a 空 
q: 问题
a: 回答

注意事项:
a字段：对于工具，代码执行 格式为{metadata}\n{content} , metadata 一般为函数名或者 interpreter，工具函数例如: 函数名\n{content}  或者 代码执行: interpreter\n{content}
详细参考(assets/PROMPT.md),(assets/TOOLS.md)
a字段：对于普通聊天，a即为回答。
```

```json
{"id": 1, "paragraph": [{"role": "system", "q": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.", "a": ""}, {"role": "user", "q": "Hello", "a": "Hello, I'm ChatGLM3. What can I assist you today?"}]}
{"id": 2, "paragraph": [{"role": "system", "q": "Answer the following questions as best as you can. You have access to the following tools:\n[\n    {\n        \"name\": \"get_current_weather\",\n        \"description\": \"Get the current weather in a given location\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"location\": {\n                    \"type\": \"string\",\n                    \"description\": \"The city and state, e.g. San Francisco, CA\",\n                },\n                \"unit\": {\"type\": \"string\"},\n            },\n            \"required\": [\"location\"],\n        },\n    }\n]\n", "a": ""}, {"role": "user", "q": "今天北京的天气怎么样？", "a": "好的，让我们来查看今天的天气\n<|assistant|>get_current_weather\n```python\ntool_call(location=\"beijing\", unit=\"celsius\")\n```\n"}, {"role": "observation", "q": "{\"temperature\": 22}", "a": "根据查询结果，今天北京的气温为 22 摄氏度。"}]}
{"id": 3, "paragraph": [{"role": "system", "q": "你是一位智能AI助手，你叫ChatGLM3，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/", "a": ""}, {"role": "user", "q": "#File: /mnt/data/metadata.jsonl\n#Size: 35380\n#File uploaded\n文件中是否存在缺失值或异常值？", "a": "首先，我将读取您提供的文件，并查看其内容以确定是否存在缺失值或异常值。\n\n我们从加载文件开始：\n<|assistant|>interpreter\n```python\nimport json\n\n# Load the uploaded file\ndata = []\nwith open(\"/mnt/data/metadata.jsonl\", 'r') as file:\n    for line in file:\n        data.append(json.loads(line))\n\n# Display the first few records to get a sense of the data\ndata[:5]\n```\n"}, {"role": "observation", "q": "\n[{'file_name': 'aceinthehole.png',\n  'name': 'Ace in the Hole',\n  'type': 'survivor',\n  'description': 'Lady Luck always seems to be throwing something good your way.'},\n {'file_name': 'adrenaline.png',\n  'name': 'Adrenaline',\n  'type': 'survivor',\n  'description': 'You are fuelled by unexpected energy when on the verge of escape.'},\n {'file_name': 'aftercare.png',\n  'name': 'Aftercare',\n  'type': 'survivor',\n  'description': \"Unlocks potential in one's aura reading ability.\"},\n {'file_name': 'agitation.png',\n  'name': 'Agitation',\n  'type': 'killer',\n  'description': 'You get excited in anticipation of hooking your prey.'},\n {'file_name': 'alert.png',\n  'name': 'Alert',\n  'type': 'survivor',\n  'description': 'Your acute senses are on high alert.'}]\n        ", "a": "\n该文件看起来包含有关某些条目的元数据，每个条目有以下字段：\n\n- `file_name`: 文件名称\n- `name`: 名称\n- `type`: 类型（例如 \"survivor\" 或 \"killer\"）\n- `description`: 描述\n\n接下来，我将检查这些字段是否存在缺失值。\n<|assistant|>interpreter\n```python\n# Check for missing values in each column\nmissing_values = {key: sum(1 for item in data if key not in item) for key in data[0].keys()}\n\nmissing_values\n```\n"}, {"role": "observation", "q": "{'file_name': 0, 'name': 0, 'type': 0, 'description': 0}", "a": "\n每个字段都没有缺失值。这意味着每个条目都完整。\n\n接下来，我们可以检查是否存在任何潜在的异常值。这通常需要更深入的分析和对数据的了解，但我可以从一些基本的统计和分布开始进行检查。为了这样做，我们可以查看“type”字段的值分布，看看是否有任何意外的类别。\n<|assistant|>interpreter\n```python\n# Check the distribution of values in the 'type' column\ntype_distribution = {key: sum(1 for item in data if item['type'] == key) for key in set(item['type'] for item in data)}\n\ntype_distribution\n            "}, {"role": "observation", "q": "{'survivor': 116, 'killer': 103}", "a": "\n“type”字段有两个值：“survivor”和“killer”。其中，“survivor”有116个条目，而“killer”有103个条目。\n\n从这个初步的检查中，我们没有发现任何明显的异常值。\n\n如果您有任何特定的标准或其他字段需要检查，我可以为您提供更详细的分析。您还希望我检查什么呢？\n             "}, {"role": "user", "q": "帮我画一个爱心", "a": "interpreter\n```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# 爱心形状的参数方程\ndef heart(t):\n    x = 16 * np.sin(t) ** 3\n    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)\n    return x, y\n\nt = np.linspace(0, 2 * np.pi, 1000)\nx, y = heart(t)\n\nplt.figure(figsize=(6, 6))\nplt.plot(x, y, color='red')\nplt.axis('equal')\nplt.axis('off')\nplt.show()\n```\n"}, {"role": "observation", "q": "【image】", "a": "这是一个爱心形状。我使用了参数方程来描述这个形状，并使用matplotlib进行了绘制。如果您有任何其他需求或问题，请随时告诉我。"}]}
{"id": 4, "paragraph": [{"role": "user", "q": "写一个诗歌，关于冬天", "a": "冬夜寂静冷，\n云在天边飘，\n冰封白雪上， \n寒冷像一场雪。\n \n雪花融化成冰，\n像那雪花飘洒，\n在寒冷的冬天，\n感受春天的喜悦。\n 冬日里，\n风雪渐消，\n一片寂静，\n把快乐和温暖带回家。"}]}
{"id": 5, "paragraph": [{"role": "user", "q": "晚上睡不着应该怎么办", "a": "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:\n\n1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。\n2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。\n3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。\n4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。\n5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。\n6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。\n\n如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"}]}
```

或者

```json
{"id": 1, "conversations": [{"from": "system", "value": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."}, {"from": "user", "value": "Hello"}, {"from": "assistant", "value": "Hello, I'm ChatGLM3. What can I assist you today?"}]}
{"id": 2, "conversations": [{"from": "system", "value": "Answer the following questions as best as you can. You have access to the following tools:\n[\n    {\n        \"name\": \"get_current_weather\",\n        \"description\": \"Get the current weather in a given location\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"location\": {\n                    \"type\": \"string\",\n                    \"description\": \"The city and state, e.g. San Francisco, CA\",\n                },\n                \"unit\": {\"type\": \"string\"},\n            },\n            \"required\": [\"location\"],\n        },\n    }\n]\n"}, {"from": "user", "value": "今天北京的天气怎么样？"}, {"from": "assistant", "value": "好的，让我们来查看今天的天气\n<|assistant|>get_current_weather\n```python\ntool_call(location=\"beijing\", unit=\"celsius\")\n```\n"}, {"from": "observation", "value": "{\"temperature\": 22}"}, {"from": "assistant", "value": "根据查询结果，今天北京的气温为 22 摄氏度。"}]}
{"id": 3, "conversations": [{"from": "system", "value": "你是一位智能AI助手，你叫ChatGLM3，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/"}, {"from": "user", "value": "#File: /mnt/data/metadata.jsonl\n#Size: 35380\n#File uploaded\n文件中是否存在缺失值或异常值？"}, {"from": "assistant", "value": "首先，我将读取您提供的文件，并查看其内容以确定是否存在缺失值或异常值。\n\n我们从加载文件开始：\n<|assistant|>interpreter\n```python\nimport json\n\n# Load the uploaded file\ndata = []\nwith open(\"/mnt/data/metadata.jsonl\", 'r') as file:\n    for line in file:\n        data.append(json.loads(line))\n\n# Display the first few records to get a sense of the data\ndata[:5]\n```\n"}, {"from": "observation", "value": "\n[{'file_name': 'aceinthehole.png',\n  'name': 'Ace in the Hole',\n  'type': 'survivor',\n  'description': 'Lady Luck always seems to be throwing something good your way.'},\n {'file_name': 'adrenaline.png',\n  'name': 'Adrenaline',\n  'type': 'survivor',\n  'description': 'You are fuelled by unexpected energy when on the verge of escape.'},\n {'file_name': 'aftercare.png',\n  'name': 'Aftercare',\n  'type': 'survivor',\n  'description': \"Unlocks potential in one's aura reading ability.\"},\n {'file_name': 'agitation.png',\n  'name': 'Agitation',\n  'type': 'killer',\n  'description': 'You get excited in anticipation of hooking your prey.'},\n {'file_name': 'alert.png',\n  'name': 'Alert',\n  'type': 'survivor',\n  'description': 'Your acute senses are on high alert.'}]\n        "}, {"from": "assistant", "value": "\n该文件看起来包含有关某些条目的元数据，每个条目有以下字段：\n\n- `file_name`: 文件名称\n- `name`: 名称\n- `type`: 类型（例如 \"survivor\" 或 \"killer\"）\n- `description`: 描述\n\n接下来，我将检查这些字段是否存在缺失值。\n<|assistant|>interpreter\n```python\n# Check for missing values in each column\nmissing_values = {key: sum(1 for item in data if key not in item) for key in data[0].keys()}\n\nmissing_values\n```\n"}, {"from": "observation", "value": "{'file_name': 0, 'name': 0, 'type': 0, 'description': 0}"}, {"from": "assistant", "value": "\n每个字段都没有缺失值。这意味着每个条目都完整。\n\n接下来，我们可以检查是否存在任何潜在的异常值。这通常需要更深入的分析和对数据的了解，但我可以从一些基本的统计和分布开始进行检查。为了这样做，我们可以查看“type”字段的值分布，看看是否有任何意外的类别。\n<|assistant|>interpreter\n```python\n# Check the distribution of values in the 'type' column\ntype_distribution = {key: sum(1 for item in data if item['type'] == key) for key in set(item['type'] for item in data)}\n\ntype_distribution\n            "}, {"from": "observation", "value": "{'survivor': 116, 'killer': 103}"}, {"from": "assistant", "value": "\n“type”字段有两个值：“survivor”和“killer”。其中，“survivor”有116个条目，而“killer”有103个条目。\n\n从这个初步的检查中，我们没有发现任何明显的异常值。\n\n如果您有任何特定的标准或其他字段需要检查，我可以为您提供更详细的分析。您还希望我检查什么呢？\n             "}, {"from": "user", "value": "帮我画一个爱心"}, {"from": "assistant", "value": "interpreter\n```python\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# 爱心形状的参数方程\ndef heart(t):\n    x = 16 * np.sin(t) ** 3\n    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)\n    return x, y\n\nt = np.linspace(0, 2 * np.pi, 1000)\nx, y = heart(t)\n\nplt.figure(figsize=(6, 6))\nplt.plot(x, y, color='red')\nplt.axis('equal')\nplt.axis('off')\nplt.show()\n```\n"}, {"from": "observation", "value": "【image】"}, {"from": "assistant", "value": "这是一个爱心形状。我使用了参数方程来描述这个形状，并使用matplotlib进行了绘制。如果您有任何其他需求或问题，请随时告诉我。"}]}
{"id": 4, "conversations": [{"from": "user", "value": "写一个诗歌，关于冬天"}, {"from": "assistant", "value": "冬夜寂静冷，\n云在天边飘，\n冰封白雪上， \n寒冷像一场雪。\n \n雪花融化成冰，\n像那雪花飘洒，\n在寒冷的冬天，\n感受春天的喜悦。\n 冬日里，\n风雪渐消，\n一片寂静，\n把快乐和温暖带回家。"}]}
{"id": 5, "conversations": [{"from": "user", "value": "晚上睡不着应该怎么办"}, {"from": "assistant", "value": "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:\n\n1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。\n2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。\n3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。\n4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。\n5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。\n6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。\n\n如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"}]}
```


## aigc-serving

部署chatglm3之后 ， 可以通过 
- [test_chatglm3_tools.py](https://github.com/ssbuild/aigc_serving/blob/main/tests/test_chatglm3_tools.py)
- [test_chatglm3_code.py](https://github.com/ssbuild/aigc_serving/blob/main/tests/test_chatglm3_code.py)

体验tools 和 code 功能

## infer
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理lora微调模型
     python infer.py


| **量化等级**    | **最低 GPU 显存** |
| -------------- | ----------------- |
| FP16（无量化）   | 13 GB             |
| INT8           | 10 GB              |
| INT4           | 6 GB               |





## training
```text
# 制作数据
cd scripts
bash train_full.sh -m dataset 
or
bash train_lora.sh -m dataset 
or
bash train_ptv2.sh -m dataset 

注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)

# 全参数训练 
    bash train_full.sh -m train 
    
# lora adalora ia3 
    bash train_lora.sh -m train 
    
# ptv2
    bash train_ptv2.sh -m train 
```

## 训练参数
[训练参数](assets/args.MD)




## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [moss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [chatglm3_finetuning](https://github.com/ssbuild/chatglm3_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [skywork_finetuning](https://github.com/ssbuild/skywork_finetuning)
- [bluelm_finetuning](https://github.com/ssbuild/bluelm_finetuning)
- [yi_finetuning](https://github.com/ssbuild/yi_finetuning)
- [aigc_evals](https://github.com/ssbuild/aigc_evals)
- [aigc_serving](https://github.com/ssbuild/aigc_serving)


## 
    纯粹而干净的代码


    

## Reference
    https://huggingface.co/THUDM/chatglm3-6b




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/chatglm3_finetuning&type=Date)](https://star-history.com/#ssbuild/chatglm3_finetuning&Date)

