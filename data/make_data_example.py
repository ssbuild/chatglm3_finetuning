# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50


import json

# 对话例子
x0 = [
    {
        "role": "system",
        "metadata": "",
        "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
    },
    {
        "role": "user",
        "metadata": "",
        "content": "Hello"
    },
    {
        "role": "user",
        "metadata": "",
        "content": "Hello, I'm ChatGLM3. What can I assist you today?"
    },
]

#工具例子

x1 = [
    {
        "role": "system",
        "metadata": "",
        "content": """Answer the following questions as best as you can. You have access to the following tools:\n""",
        "tools": [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]
    },
    {
        "role": "user",
        "metadata": "",
        "content": "今天北京的天气怎么样？"
    },
    {
        "role": "assistant",
        "metadata": "",
        "content": "好的，让我们来查看今天的天气"
    },
    {
        "role": "assistant",
        "metadata": "get_current_weather",
        "content": """```python
tool_call(location="beijing", unit="celsius")
```
<|observation|>
{"temperature": 22}"""
    },
    {
        "role": "assistant",
        "metadata": "",
        "content": "根据查询结果， 今天北京的气温为22 摄氏度。"
    },
]


# 代码执行

x2 = [
    {
        "role": "system",
        "metadata": "",
        "content": "你是一位智能AI助手，你叫ChatGLM3，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。"
    },
    {
        "role": "user",
        "metadata": "",
        "content": """#File: /mnt/data/metadata.jsonl
#Size: 35380
#File uploaded
文件中是否存在缺失值或异常值？"""
    },
    {
        "role": "assistant",
        "metadata": "interpreter",
        "content": """```python
import json

# Load the uploaded file
data = []
with open("/mnt/data/metadata.jsonl", 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Display the first few records to get a sense of the data
data[:5]
```
<|observation|>
```result
[{'file_name': 'aceinthehole.png',
  'name': 'Ace in the Hole',
  'type': 'survivor',
  'description': 'Lady Luck always seems to be throwing something good your way.'},
 {'file_name': 'adrenaline.png',
  'name': 'Adrenaline',
  'type': 'survivor',
  'description': 'You are fuelled by unexpected energy when on the verge of escape.'},
 {'file_name': 'aftercare.png',
  'name': 'Aftercare',
  'type': 'survivor',
  'description': "Unlocks potential in one's aura reading ability."},
 {'file_name': 'agitation.png',
  'name': 'Agitation',
  'type': 'killer',
  'description': 'You get excited in anticipation of hooking your prey.'},
 {'file_name': 'alert.png',
  'name': 'Alert',
  'type': 'survivor',
  'description': 'Your acute senses are on high alert.'}]
```
"""
    },
    {
        "role": "assistant",
        "metadata": "该文件看起来包含有关某些条目的元数据，每个条目有以下字段：",
        "content": """
- `file_name`: 文件名称
- `name`: 名称
- `type`: 类型（例如 "survivor" 或 "killer"）
- `description`: 描述

接下来，我将检查这些字段是否存在缺失值。"""
    },
    {
        "role": "assistant",
        "metadata": "interpreter",
        "content": """```python
# Check for missing values in each column
missing_values = {key: sum(1 for item in data if key not in item) for key in data[0].keys()}

missing_values
```
<|observation|>
```result
{'file_name': 0, 'name': 0, 'type': 0, 'description': 0}
```"""
    },
    {
        "role": "assistant",
        "metadata": "每个字段都没有缺失值。这意味着每个条目都完整。",
        "content": """
接下来，我们可以检查是否存在任何潜在的异常值。这通常需要更深入的分析和对数据的了解，但我可以从一些基本的统计和分布开始进行检查。为了这样做，我们可以查看“type”字段的值分布，看看是否有任何意外的类别。
"""
    },
    {
        "role": "assistant",
        "metadata": "interpreter",
        "content": """```python
# Check the distribution of values in the 'type' column
type_distribution = {key: sum(1 for item in data if item['type'] == key) for key in set(item['type'] for item in data)}

type_distribution
```
<|observation|>
```result
{'survivor': 116, 'killer': 103}
```"""
    },
    {
        "role": "assistant",
        "metadata": "“type”字段有两个值：“survivor”和“killer”。其中，“survivor”有116个条目，而“killer”有103个条目。",
        "content": """
从这个初步的检查中，我们没有发现任何明显的异常值。

如果您有任何特定的标准或其他字段需要检查，我可以为您提供更详细的分析。您还希望我检查什么呢？"""
    },
    {
        "role": "user",
        "metadata": "",
        "content": """帮我画一个爱心"""
    },
    {
        "role": "assistant",
        "metadata": "interpreter",
        "content": """```python
import numpy as np
import matplotlib.pyplot as plt

# 爱心形状的参数方程
def heart(t):
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x, y

t = np.linspace(0, 2 * np.pi, 1000)
x, y = heart(t)

plt.figure(figsize=(6, 6))
plt.plot(x, y, color='red')
plt.axis('equal')
plt.axis('off')
plt.show()
```
<|observation|>
```result
【image】
```"""
    },
    {
        "role": "assistant",
        "metadata": "",
        "content": """这是一个爱心形状。我使用了参数方程来描述这个形状，并使用matplotlib进行了绘制。如果您有任何其他需求或问题，请随时告诉我。"""
    }
]



x3 = [
    {
        "role": "user",
        "metadata": "",
        "content": "写一个诗歌，关于冬天"
    },
    {
        "role": "assistant",
        "metadata": "",
        "content": '\n'.join([
            "冬夜寂静冷，",
            "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。",
            " ",
            "雪花融化成冰，",
            "像那雪花飘洒，",
            "在寒冷的冬天，",
            "感受春天的喜悦。",
            " 冬日里，",
            "风雪渐消，",
            "一片寂静，",
            "把快乐和温暖带回家。"
        ])
    },
]


x4 = [
    {
        "role": "user",
        "metadata": "",
        "content": "晚上睡不着应该怎么办"
    },
    {
        "role": "assistant",
        "metadata": "",
        "content": '\n'.join([
            "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:",
            "",
            "1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。",
            "2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。",
            "3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。",
            "4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。",
            "5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。",
            "6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。",
            "",
            "如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"
        ])
    },
]





x = [x0,x1,x2,x3,x4]

with open('./finetune_train_examples.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(100):
        for j in range(len(x)):
            index += 1
            f.write(json.dumps(x[j],ensure_ascii=False) + '\n' )
