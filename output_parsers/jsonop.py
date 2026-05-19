from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()
model = ChatMistralAI(model="mistral-small-2506")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me details about {topic}. \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "ChatGPT"})

print(result['developer'])         # {'name': 'ChatGPT', 'company': 'OpenAI', ...}
print(type(result))   # <class 'dict'>