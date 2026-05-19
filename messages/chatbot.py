from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()  
model=ChatMistralAI(model="mistral-small-2506")

messages = [
    SystemMessage(content="You are a helpful assistant named Bob.")  # sets AI personality
]
print("------------welcome to chatbot-------------")
while True:
    prompt=input("you----")
    if prompt=='0':
        break
    messages.append(HumanMessage(content=prompt))   # your message
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))  # AI's reply saved
    print("bob ---", response.content)


