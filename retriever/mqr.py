from langchain_core.documents import Document
from  langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
# Relevant health & wellness documents
docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize hugging face  embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# LLM
llm = ChatHuggingFace(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
)

# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Create retrievers
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
   llm=llm
)

# Query
query = "How to improve energy levels and maintain balance?"

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results= multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# --------------------------notes ------------
# MultiQueryRetriever (MQR) - Short Notes

# Definition:
# MultiQueryRetriever improves retrieval by generating multiple variations of a user query using an LLM and combining results.

# Purpose:
# - Overcome limitation of single-query retrieval
# - Capture different meanings/aspects of a query
# - Improve recall (find more relevant docs)

# How it works:
# 1. User query → sent to LLM
# 2. LLM generates multiple rephrased queries
# 3. Each query → run through retriever
# 4. Combine all results
# 5. Remove duplicates
# 6. Return final set of documents

# Key Idea:
# "Different queries → better coverage"

# Example:
# Query: "How to improve energy?"

# Generated queries:
# - How to increase energy levels?
# - Ways to stay energetic daily?
# - Tips for boosting metabolism?

# Advantages:
# - Better recall than similarity search
# - Handles ambiguous queries
# - Finds hidden relevant documents

# Disadvantages:
# - Slower (multiple queries)
# - Higher cost (more LLM calls)

# LangChain Usage:
# from langchain.retrievers.multi_query import MultiQueryRetriever

# multiquery_retriever = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=llm
# )

# results = multiquery_retriever.invoke(query)

# Use Cases:
# - RAG systems
# - Chatbots
# - Search systems
# - AI agents

# Comparison:
# - Similarity Search → 1 query → limited results
# - MMR → reduces redundancy
# - MQR → expands query → better coverage

# One-line:
# MQR = Multiple queries → better retrieval results