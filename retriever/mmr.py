from langchain_core.documents import Document
from  langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]


# Initialize hugging face  embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)
# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# --------------------notes -----------------
# MMR (Max Marginal Relevance) - Short Notes

# Definition:
# MMR is a retrieval technique used to get results that are both relevant to the query and diverse among themselves.

# Purpose:
# - Avoid duplicate / similar results
# - Improve information coverage
# - Balance relevance + diversity

# Working Idea:
# Score = Relevance(query, doc) - Redundancy(doc, selected_docs)

# Key Parameters:
# - k → number of results to return
# - lambda_mult → controls balance
#     1.0 → only relevance (like normal search)
#     0.0 → only diversity
#     0.5 → balanced (recommended)

# Without MMR:
# - Top results are very similar
# - Redundant information

# With MMR:
# - Results are different from each other
# - Covers multiple aspects of query

# Use Cases:
# - RAG (Retrieval-Augmented Generation)
# - Chatbots
# - Search systems
# - AI agents

# LangChain Usage:
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 3, "lambda_mult": 0.5}
# )

# One-line:
# MMR = Relevant + Diverse results (no repetition)