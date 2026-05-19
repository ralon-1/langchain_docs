from langchain_mistralai import ChatMistralAI
from typing import Annotated,Literal,Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()  
model=ChatMistralAI(model="mistral-small-2506")

#   make schema
class review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]


structured_model=model.with_structured_output(review)
result=structured_model.invoke("The movie was fantastic! The storyline was gripping and the acting was top-notch. However, the pacing was a bit slow in the middle. Overall, I would highly recommend it to anyone who loves a good thriller.")

print("Key Themes:", result['key_themes'])
print("Summary:", result['summary'])
print("Sentiment:", result['sentiment'])
print("Pros:", result['pros'])
print("Cons:", result['cons'])
print("Name:", result['name'])