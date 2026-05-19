from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
loader=DirectoryLoader(
    path="document_loaders/pdfs",
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
docs=loader.lazy_load()

# print(docs[12].page_content)
for doc in docs:
 print(doc.page_content)


# ///////// load vs lazy load
# ================================

# load() vs lazy_load() in LangChain

# ================================

# 🔹 1. load()

# 👉 Definition:

# load() reads the ENTIRE file at once and returns all documents.

# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("file.pdf")

# docs = loader.load()   # loads everything

# docs = [Document1, Document2, Document3, ...]

# --------------------------------

# 🔹 Key Points of load()

# --------------------------------

# ✅ Loads full data into memory

# ✅ Returns a LIST

# ❌ Not memory efficient for large files

# ✅ Easy to use

# 🔹 2. lazy_load()

# 👉 Definition:

# lazy_load() loads documents ONE BY ONE (on demand).

# loader = PyPDFLoader("file.pdf")

# docs = loader.lazy_load()   # generator (not list)

# docs is NOT a list, it is a generator

# --------------------------------

# 🔹 How to use lazy_load()

# --------------------------------

# for doc in docs:
# print(doc.page_content)

# --------------------------------

# 🔹 Key Points of lazy_load()

# --------------------------------

# ✅ Loads data page-by-page

# ✅ Memory efficient

# ❌ Slightly more complex to use

# ❌ Cannot use indexing like docs[0]

# 🔹 3. Difference Table

# --------------------------------

# load()                     lazy_load()

# --------------------------------------------

# Returns list              Returns generator

# Loads all at once         Loads one by one

# High memory usage         Low memory usage

# Easy indexing             No indexing

# Good for small files      Best for large files

# 🔹 4. Real-world analogy

# load()      → Download entire movie 🎬

# lazy_load() → Stream movie (buffering) 📡

# 🔹 5. When to use what?

# Use load():

# - Small files

# - Quick prototyping

# - When memory is not an issue

# Use lazy_load():

# - Large PDFs

# - Production systems

# - RAG pipelines

# - Memory optimization needed

# 🔹 6. Important Interview Point

# lazy_load() returns a GENERATOR

# Example:

docs = loader.lazy_load()
print(type(docs))

# Output:

# <class 'generator'>

# 🔥 FINAL SUMMARY

# load():

# → loads everything at once

# → returns list

# lazy_load():

# → loads one-by-one

# → returns generator

# → memory efficient

# ================================

# END

# ================================
