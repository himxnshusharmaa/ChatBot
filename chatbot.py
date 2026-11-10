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

print("choose your AI model")
print("press 1 for angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")

choice = int(input("tell your response: "))

if choice == 1:
    mode = "you are an angry AI agent. you are very rude and aggressive in your responses."
elif choice == 2:
    mode = "you are a funny AI agent. you are very humorous and witty in your responses."
elif choice == 3:   
    mode = "you are a sad AI agent. you are very emotional and melancholic in your responses."
messages = [SystemMessage(content=mode)]

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
