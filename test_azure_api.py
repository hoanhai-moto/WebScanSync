import os
from openai import AzureOpenAI

endpoint = "https://websc-mb7cmyp6-eastus2.cognitiveservices.azure.com/"
model_name = "model-router"
deployment = "model-router"

subscription_key = "CxHy8XVccoHfZZStVm0CVF1uJgiDqnHCdDz7S6rxrQngMqsXdXxPJQQJ99BEACHYHv6XJ3w3AAAAACOGECqA"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=10000,
    model=deployment
)

print(response.choices[0].message.content)

