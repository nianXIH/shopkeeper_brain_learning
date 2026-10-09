from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# 初始化OpenAI客户端
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg"
                    },
                },
                {"type": "text", "text": "这道题怎么解答？"},
            ]
        }
    ]
)

print(completion.choices[0].message.content)