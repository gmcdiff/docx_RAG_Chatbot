# Install required packages

#!pip install langchain-community chromadb unstructured python-docx huggingface_hub

# --- Imports ---
import os
os.environ["CHROMA_TELEMETRY"] = "0"
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import chromadb
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# --- Load DOCX Files ---

def extract_text_from_docx_in_directory(directory_path):

    all_documents = []

    file_names = []


    for filename in os.listdir(directory_path):

        if filename.lower().endswith(".docx"):

            print(f"Loading: {filename}")

            file_path = os.path.join(directory_path, filename)

            loader = UnstructuredWordDocumentLoader(file_path)

            docs = loader.load()

            all_documents.extend(docs)

            file_names.append(filename)


    return all_documents, file_names


all_documents, file_names = extract_text_from_docx_in_directory("documents")


# --- Initialize ChromaDB ---

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="docx_collection")


collection.add(

    documents=[doc.page_content for doc in all_documents],

    metadatas=[{'id': f} for f in file_names],

    ids=file_names

)


print(f"Total documents in collection: {collection.count()}")


# --- Semantic Search ---

def get_ss_results_text(query, collection, n_results):

    ss_result = collection.query(

        query_texts=[query],

        n_results=n_results,

    )

    final_result_string = "\n\n".join(ss_result["documents"][0])

    return final_result_string, ss_result


# --- Hugging Face LLM Setup ---

# Use your own token here or set it as an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient("deepseek-ai/DeepSeek-V3", token=HF_TOKEN)


def llm_invoke(messages):

    response = client.chat_completion(

        messages=messages,

        max_tokens=512

    )

    return response['choices'][0]['message']['content']


# --- RAG Chatbot Loop ---

def rag_chatbot():

    conversation_memory = [{"role": "system", "content": "You are a helpful assistant."}]

    print("Welcome to the RAG Chatbot! Type '\\quit' to exit.\n")


    while True:

        user_query = input("User: ")

        if user_query.strip().lower() == "\\quit":

            print("Exiting the chatbot. Goodbye!")

            break

        context_string, docs = get_ss_results_text(user_query, collection, n_results=1)

        context_message = f"Relevant context:\n{context_string}"

        conversation_memory.append({"role": "user", "content": f"{context_message}\n\n{user_query}"})

        answer = llm_invoke(conversation_memory)

        conversation_memory.append({"role": "assistant", "content": answer})

        print(f"Assistant: {answer}\n")

# --- Run the chatbot ---

rag_chatbot()