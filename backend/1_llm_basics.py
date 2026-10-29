from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.environ["NVIDIA_API_KEY"]
)

# result = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-lightning-30b-a3b",
#     messages=[{"role":"user","content":"What is the square root of 49?"}],
# )

# print(json.dumps(result.to_dict(), indent=2))

# result = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-lightning-30b-a3b",
#     messages=[
#         # {"role": "user", "content": "My name is Harish"},
#         # {"role": "assistant", "content": "Nice to meet you Harish!"},
#         {"role": "user", "content": "What is my name?"}
#     ]
# )

# print(json.dumps(result.to_dict(), indent=2))

# result = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-lightning-30b-a3b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant who makes jokes."},
#         {"role": "user", "content": "What is the square root of 49?"},
#     ],
#     temperature=0.7,
#     max_tokens=100,
# )

# print(json.dumps(result.to_dict(), indent=2))

# # ---- 6. Token usage ----
# print(f"Input tokens: {result.usage.prompt_tokens}")
# print(f"Output tokens: {result.usage.completion_tokens}")
# print(f"Total tokens: {result.usage.total_tokens}")

# ---- 7. Streaming ----
stream = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short story about a robot."},
    ],
    stream=True
)

for event in stream:
    if event.choices:
        delta = event.choices[0].delta
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            print(reasoning_content, end="", flush=True)
