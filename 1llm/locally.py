from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
     model_kwargs={"token": os.getenv("HUGGINGFACEHUB_API_TOKEN")}  # add this
)

model=ChatHuggingFace(llm=llm)
ans=model.invoke("who are you")
print(ans.content)