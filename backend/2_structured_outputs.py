from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json

load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.environ["NVIDIA_API_KEY"]
)

# ---- 1. The problem - unstructured output ----
# result = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-lightning-30b-a3b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Give me the name, age and city of a fictional person."},
#     ],
# )

# print(json.dumps(result.to_dict(), indent=2))

# print(type(result.output_text))
# Output: "Sure! Here's a fictional person: Name: John Smith, Age: 32, City: Chicago"
# You can't reliably parse this in your code

# ---- 2. Naive fix - ask for JSON in the prompt ----

# result = client.chat.completions.create(
#     model="nvidia/nemotron-3.5-lightning-30b-a3b",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant. Respond in JSON format."},
#         {"role": "user", "content": "Give me the name, age and city of a fictional person."},
#     ],
# )

# print(json.dumps(result.to_dict(), indent=2))
# # Sometimes works, but the model might wrap it in ```json ``` blocks
# # or change the key names, or add extra fields
# # Not reliable enough for production



# ---- What is Pydantic? ----
# Pydantic lets you define the exact "shape" of data you expect.
# Each field has a name and a type (str, int, float, bool, list, etc.)
#
# When you pass response_format=Person to chat.completions.parse, the SDK
# converts this class into a JSON schema. The endpoint must support structured
# outputs for the schema to be enforced by the model.
#
# This is the key difference:
# "Respond in JSON" in the prompt = a suggestion the model can ignore
# response_format=Person = schema-constrained output when supported by the endpoint

# ---- 3. Structured output with Pydantic ----
class Person(BaseModel):
    name: str
    age: int
    city: str

result = client.chat.completions.parse(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me the name, age and city of a fictional person."},
    ],
    response_format=Person,
)

# print(json.dumps(result.to_dict(), indent=2))

person = result.choices[0].message.parsed
print(person)
print(person.name)
print(person.age)
print(person.city)