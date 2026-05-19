from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
text=[
    "The sentence-transformers/all-MiniLM-L6-v2 i",
   " s a lightweight and efficient AI model available on",
     " HuggingFace, designed specifically for generating s"
    #   entence embeddings. It converts raw text into high-dimensional numerical vectors that capture the semantic meaning of sentences. These vectors are extremely useful in tasks like semantic search, where you want to find the most relevant documents or sentences based on meaning rather than exact keyword matching. The model is very popular among developers because it is small in size, fast to run, and does not require a powerful GPU to work efficiently. It is widely used in real-world applications like chatbots, recommendation systems, and Retrieval-Augmented Generation (RAG) pipelines where understanding the context of text is crucial.
]
vector=embedding.embed_documents(text)
print(vector)