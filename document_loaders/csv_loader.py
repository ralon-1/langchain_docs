from langchain_community.document_loaders import CSVLoader
loader=CSVLoader(file_path='document_loaders/social_data.csv')
docs=loader.load()
print(len(docs))
print(docs[2].page_content)