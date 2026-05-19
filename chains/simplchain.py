from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()
template = PromptTemplate(
    template="Give me details about {topic}",
    input_variables=["topic"]
)

chain = template | model | parser

result = chain.invoke({"topic": "ChatGPT"})

print(result)         