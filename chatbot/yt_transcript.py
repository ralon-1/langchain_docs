
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

transcript = [
    {'text': 'This video explains how large language models work in a simple way.', 'start': 0.0, 'duration': 4.5},
    {'text': 'They are trained on massive datasets of text collected from the internet.', 'start': 4.5, 'duration': 4.2},
    {'text': 'The model learns patterns by predicting the next word repeatedly.', 'start': 8.7, 'duration': 3.8},
    {'text': 'This process adjusts millions of parameters to improve accuracy.', 'start': 12.5, 'duration': 4.1},
    {'text': 'Instead of memorizing exact sentences, it learns relationships between words.', 'start': 16.6, 'duration': 4.3},
    {'text': 'Now let us understand how text is processed inside the model.', 'start': 20.9, 'duration': 3.6},
    {'text': 'Text is broken into smaller units called tokens before processing.', 'start': 24.5, 'duration': 4.0},
    {'text': 'Each token is converted into numbers so the model can understand it.', 'start': 28.5, 'duration': 4.2},
    {'text': 'These numerical representations are known as embeddings.', 'start': 32.7, 'duration': 3.5},
    {'text': 'Embeddings help capture semantic meaning between different words.', 'start': 36.2, 'duration': 4.0},
    {'text': 'Next comes the transformer architecture which powers modern AI models.', 'start': 40.2, 'duration': 4.5},
    {'text': 'It uses attention mechanisms to focus on important words in a sentence.', 'start': 44.7, 'duration': 5.0},
    {'text': 'Attention allows the model to understand context across long sequences.', 'start': 49.7, 'duration': 4.2},
    {'text': 'This is what makes transformers more powerful than older techniques.', 'start': 53.9, 'duration': 4.3},
    {'text': 'During training the model keeps improving its predictions step by step.', 'start': 58.2, 'duration': 4.1},
    {'text': 'It learns which words are likely to follow others in different contexts.', 'start': 62.3, 'duration': 4.4},
    {'text': 'Once trained the model can generate human like responses to prompts.', 'start': 66.7, 'duration': 4.0},
    {'text': 'A prompt is simply the input text given by the user.', 'start': 70.7, 'duration': 3.6},
    {'text': 'The model processes the prompt and predicts the most likely continuation.', 'start': 74.3, 'duration': 4.3},
    {'text': 'This is how chatbots and AI assistants generate answers.', 'start': 78.6, 'duration': 3.8},
    {'text': 'However the model does not truly understand meaning like humans do.', 'start': 82.4, 'duration': 4.2},
    {'text': 'It is only making statistically likely predictions based on training data.', 'start': 86.6, 'duration': 4.0}
]

transcript_text = " ".join(chunk["text"] for chunk in transcript)
#  step aplitting
splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.create_documents([transcript_text])


#  1c gen embeddings

# embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# vector DB
vector_store = FAISS.from_documents(chunks, embeddings)
ids = list(vector_store.index_to_docstore_id.values())

# print(vector_store.get_by_ids([ids[0]]))
#  retriver
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever.invoke("What is embedding?"))