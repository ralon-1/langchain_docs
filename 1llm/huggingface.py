from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm=HuggingFaceEndpoint(
    repo_id="NYTK/text-generation-news-gpt2-small-hungarian"
)
model=ChatHuggingFace(llm=llm)
response=model.invoke("who are you ")
print(response.content)