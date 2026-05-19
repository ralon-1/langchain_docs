from langchain_core.tools import tool
# Step 1 - create a function

def multiply(a,b):
    """multiply two number"""
    return a*b

# Step 2 - add type hints
def multiply(a:int,b:int)->int:
    """multiply two number"""
    return a*b

# Step 3 - add tool decorator
@tool
def multiply(a:int,b:int)->int:
    """multiply two number"""
    return a*b
 

result = multiply.invoke({"a":3, "b":5})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
