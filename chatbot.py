import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Let the .env value replace any old PowerShell value.
load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from the .env file.")

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq",
    api_key=api_key,
    temperature=0.7,
)

messages = [SystemMessage(content="You are a helpful assistant.")]

print("Welcome! Type 0 to exit.")

while True:
    prompt = input("You: ").strip()

    if prompt == "0":
        break

    if not prompt:
        continue

    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)

print(messages)