from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader reads a PDF file and splits it into pages,
# where each page becomes a Document.
loader=PyPDFLoader("document_loaders/dl-curriculum.pdf")
docs=loader.load()
# docs is a LIST of Document objects
# print(type(docs)) # <class 'list'>
# print(len(docs)) # number of pages in PDF
# Each element = one page of PDF
print(docs[0].page_content)
print(docs[1].metadata)