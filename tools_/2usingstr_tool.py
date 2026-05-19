from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
class multiplyinput(BaseModel):
    a:int=Field(required=True,description="the first no. to add")
    b:int=Field(required=True,description="the second no. to add")

# 
def multiply_func(a: int, b: int) -> int:
    return a * b
# /
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=multiplyinput
)

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)