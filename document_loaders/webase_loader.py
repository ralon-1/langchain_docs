
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
groq_api_key = os.getenv("GROQ_API_KEY")
# print(groq_api_key)
model=ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)
# //make a prompt template
prompt=PromptTemplate(
   template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)
parser=StrOutputParser()
url='https://www.flipkart.com/computers/storage/pen-drives/pr?sid=6bo,jdy,uar'
loader=WebBaseLoader(url)
docs=loader.load()
chain=prompt|model|parser

print(chain.invoke({'question':'What is the page is talking about?', 'text':docs[0].page_content}))
