from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model=ChatMistralAI(model="mistral-small-2506")

# llm=HuggingFaceEndpoint(
#     model="mistral-small-2506",
#        task="text-generation"
    
#             )

# model=ChatHuggingFace(llm=llm)
# first prompt detailed report
template1=PromptTemplate(
    template='write a detailed report on the ${topic}',
    input_variables=['topic']
)

# seccond prompt for summary
template2=PromptTemplate(
    template='write a 5 word summary on the ${text}',
    input_variables=['text']
)

parser=StrOutputParser()
chain=template1 | model | parser | template2 | model | parser

# prompt1=template1.invoke({"topic":"current state of AI research"})
# result=model.invoke(prompt1)
# prompt2=template2.invoke({'text':result.content})
# summary=model.invoke(prompt2)

result=chain.invoke({"topic":"current state of AI research"})
print(result)