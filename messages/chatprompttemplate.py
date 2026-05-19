from langchain_core.prompts import ChatPromptTemplate
chat=ChatPromptTemplate([
    ("system","You are a helpful assistant named {name}."),
    ("human","What is the capital of {city}?")
])
prompt=chat.invoke({'name':'Bob','city':'India'})
print(prompt)