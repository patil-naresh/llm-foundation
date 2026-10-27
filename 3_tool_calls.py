from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.environ["NVIDIA_API_KEY"]
)


# ---- 0. The problem - model has no access to real-time data ----
# result = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-lightning-30b-a3b",
#     messages=[{"role":"user","content":"What is the current weather in Bangalore?"}],
# )
# if result.choices:
#     print(result.choices[0].message.content)
# Output: "I don't have access to real-time weather data."


# ---- 1. Define the actual Python function ----
def get_weather(city: str):
    return {
        "city": city,
        "temperature": 28,
        "unit": "celsius",
        "condition": "sunny"
    }

# ---- 2. Describe the function to the model (the tool definition) ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city from country india",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get weather for"
                    }
                },
                "required": ["city"]
            },
        }
    }
]


# ---- 3. Step 1 - Send the question and tools to the model ----
input_messages = [{"role": "user", "content": "What is the current weather in new york?"}]

result = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=input_messages,
    tools=tools
)

if not result.choices:
    raise RuntimeError("The model returned no choices.")

assistant_message = result.choices[0].message
input_messages.append(assistant_message.model_dump(exclude_none=True))

for tool_call in assistant_message.tool_calls or []:
    # import pdb; pdb.set_trace()
    if tool_call.function.name != "get_weather":
        raise ValueError(f"Unknown tool requested: {tool_call.function.name}")

    function_args = json.loads(tool_call.function.arguments)
    function_result = get_weather(function_args["city"])
    input_messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(function_result),
    })

if assistant_message.tool_calls:
    final_result = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=input_messages,
        tools=tools,
    )
    if not final_result.choices:
        raise RuntimeError("The model returned no final answer.")
    print(final_result.choices[0].message.content)
else:
    print(assistant_message.content)