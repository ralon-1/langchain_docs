
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
groq_api_key = os.getenv("GROQ_API_KEY")
# print(groq_api_key)
model=ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)
# //make a prompt template
prompt=PromptTemplate(
    template="give a one line answer to the question: {question}",
    input_variables=['question']
)
# //loading ytexr loader
loader=TextLoader("/Users/pravesh/Downloads/THISPC/langchain_models/document_loaders/cricket.txt",encoding='utf-8')
# the file is store in laoder
docs=loader.load()
# 👉 This does:
# Reads the file
# Converts it into a list of Document objects
print(type(docs))
# docs is a list (array)
# It contains multiple Document objects
print(len(docs))
# 👉 What it checks: 
# Number of documents inside the list
print(docs[0].page_content)
# Prints its actual text content
# print(docs[0].metadata)
# Tells where the data came from
parser=StrOutputParser()
chain=prompt|model|parser
question = "What is the capital of France?"
# print(chain.invoke({"question":question}))

# =========================================
# FINAL UNDERSTANDING (VERY IMPORTANT)
# =========================================

# docs = [
#     Document(
#         page_content="Actual text from file",
#         metadata={"source": "cricket.txt"}
#     )
# ]

# -----------------------------------------
# FLOW:
# -----------------------------------------
# Text File → TextLoader → Document → LLM

# -----------------------------------------
# KEY POINTS:
# -----------------------------------------
# 1. TextLoader reads .txt files
# 2. Output is a LIST of Document objects
# 3. Each Document has:
#       - page_content (main text)
#       - metadata (extra info)
# 4. page_content is what we send to LLM

# -----------------------------------------
# COMMON ERROR FIX:
# -----------------------------------------
# If you get encoding error → use:
# TextLoader('file.txt', encoding='utf-8')
