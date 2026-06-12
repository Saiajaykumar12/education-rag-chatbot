from annotated_types import doc
import streamlit as st
from rag_engine import load_and_split_pdfs, build_vectorstore, build_qa_chain, ask_question

st.set_page_config(
    page_title = "Study Assistant - RAG Chatbot",
    page_icon = "📚",
    layout = "centered"
)

st.title("Study Assistant")
st.caption("Upload your textbooks or course pdf's and ask questions in plain english.")

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type = "pdf",
        accept_multiple_files = True
    )
    if uploaded_files:
        if st.button("Process documents", type="primary"):
            with st.spinner("Reading and indexing your documents..."):
                try:
                    docs = load_and_split_pdfs(uploaded_files)
                    vectorstore = build_vectorstore(docs)
                    qa_chain = build_qa_chain(vectorstore)
                    st.session_state.qa_chain = qa_chain
                    st.session_state.doc_count = len(uploaded_files)
                    st.session_state.chunk_count = len(docs)
                    st.success(f"Ready. Indexed {len(docs)} chunks from {len(uploaded_files)} documents.")

                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

        if "chunk_count" in st.session_state:
            st.metric("Chunks indexed", st.session_state.chunk_count)
            st.metric("Document loaded ", st.session_state.doc_count)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents or Study materials...."):
    if "qa_chain" not in st.session_state:
        st.warning("Please upload and process your documents first using the sidebar.")
    else:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documnet..."):
                try:
                    answer, sources = ask_question(st.session_state.qa_chain, prompt)
                    st.markdown(answer)

                    if sources:
                        with st.expander("View Source sections uses to answer this"):
                            for i, doc in enumerate(sources):
                                if hasattr(doc, 'metadata'):
                                    source_file = doc.metadata.get("source_file", "Unknown")
                                    page = doc.metadata.get("page", "?")
                                else:
                                    source_file = "Unknown"
                                    page = "?"
                                st.markdown(f"**Source {i+1}** - {source_file} page{page}")
                                st.caption(doc.page_content[:400]+ "...")
                                st.divider()

                    st.session_state.messages.append(
                        {"role":"assistant",
                         "content":answer
                         })
                except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
