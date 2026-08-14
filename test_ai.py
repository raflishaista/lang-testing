import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.path.join(os.environ.get("LLM_URL", "http://10.7.1.21"), "v1"),
    api_key=os.environ.get("LLM_KEY", "")
)

response = client.chat.completions.create(
    model="qwen-35b",
    messages=[{"role": "user", "content": "Sebutkan 3 warna primer."}],
    max_tokens=500,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response.choices[0].message.content)
