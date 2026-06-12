from dotenv import load_dotenv
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_and_split_pdfs(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.split_documents(pages)
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name
        all_docs.extend(docs)
        os.unlink(tmp_path)
    return all_docs

def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def build_qa_chain(vectorstore):
    prompt_template = """You are a helpful education assistant.
Use the following context from the uploaded study material to answer the question.
If you don't know the answer from the context, say "I couldn't find this in the uploaded documents."
Always be concise and clear.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask_question(qa_chain_tuple, question):
    chain, retriever = qa_chain_tuple
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources