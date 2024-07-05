from langchain_community.llms import Ollama
from flask import Flask, request 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import glob


app = Flask(__name__)


folder_path = "db"
cached_llm = Ollama(model="aya")
embedding = FastEmbedEmbeddings()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] Answer according to provided context with Arabic language and if you do not know say so [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

def pdfPost():
    file_name = "pdf1.pdf"
    docs = []
    arr_of_files = (glob.glob("/home/ahmed/Rag/pdf/*.pdf"))
    for file in arr_of_files:
        loader =  PyPDFLoader(file)
        docs.extend(loader.load_and_split())
        
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

def startup_app():
    app.run(host="0.0.0.0", port=8080, debug= True)

if __name__ == "__main__":
    pdfPost()
    startup_app()

