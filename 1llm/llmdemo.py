
from dotenv import load_dotenv
import os
load_dotenv()  
from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.3-70b-versatile"
    # other params...
)
result=model.invoke("tell me a funny joke of 5 lines")
print(result.conten)