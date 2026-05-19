from langchain_mistralai import ChatMistralAI
from typing import Annotated,Literal,Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()  
model=ChatMistralAI(model="mistral-small-2506")

#   make schema
class review(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    summary: str = Field(description="Brief summary of the review")
    sentiment: str = Field(description="pos or neg")
    pros: Optional[list[str]] = Field(default=None, description="List of pros")
    cons: Optional[list[str]] = Field(default=None, description="List of cons")
    name: Optional[str] = Field(default="le re land ke naam wale ", description="Name of reviewer")

structured_model=model.with_structured_output(review)
result=structured_model.invoke("The movie was fantastic! The storyline was gripping and the acting was top-notch. However, the pacing was a bit slow in the middle. Overall, I would highly recommend it to anyone who loves a good thriller.")

print(result.pros)
print(result.name)