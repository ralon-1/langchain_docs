from langchain_community.tools import DuckDuckGoSearchRun
# DuckDuckGoSearchRun it is a class or tool
search_tool=DuckDuckGoSearchRun()
#  search tool is a object 
result=search_tool.invoke('solve leetcode problem ugly no. 1 ')
# print(result)
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)