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

# Mock order data for learning purposes only.
MOCK_ORDERS = {
    "A123": {
        "order_id": "A123",
        "status": "Shipped",
        "estimated_delivery": "2026-10-29",
    },
    "B456": {
        "order_id": "B456",
        "status": "Processing",
        "estimated_delivery": None,
    },
}

def get_order_status(order_id: str):
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        return {
            "order_id": order_id,
            "message": "No order is found with the provided order ID.",
            "found": False
        }
    else:
        return { "found": True, **order }



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
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status of an order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The ID of the order to check"
                    }
                },
                "required": ["order_id"]
            },
        }
    }
]


# ---- 3. Step 1 - Send the question and tools to the model ----
input_messages = [{"role": "user", "content": "Can you check order A123 and tell me its status and estimated delivery?"}]

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
    if tool_call.function.name == "get_order_status":
        function_args = json.loads(tool_call.function.arguments)
        function_result = get_order_status(function_args["order_id"])
    elif tool_call.function.name == "get_weather":
        function_args = json.loads(tool_call.function.arguments)
        function_result = get_weather(function_args["city"])
    else:
        raise ValueError(f"Unknown tool requested: {tool_call.function.name}")

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