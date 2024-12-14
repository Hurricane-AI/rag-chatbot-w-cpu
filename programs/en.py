import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from transformers import pipeline

# Helsinki-NLPのモデルを使用して日本語を英語に翻訳
translator_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
print(translator_to_en("初めまして、こんにちは。")[0]["translation_text"])

translator_to_ja = pipeline("translation", model="staka/fugumt-en-ja")

st.title("Webページとのチャット 🌐")
st.caption("このアプリでは、ローカルのLlama-3とRAGを使用してWebページとチャットすることができます")

# ユーザーからWebページのURLを取得
webpage_url = st.text_input("WebページのURLを入力してください", type="default")

if webpage_url:
    # 1. データの読み込み
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    
    # 2. Ollamaの埋め込みとベクトルストアの作成
    embeddings = OllamaEmbeddings(model="hf.co/lm-kit/bge-m3-gguf:Q5_K_M")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # 3. Ollama Llama3モデルを呼び出す
    def ollama_llm(question, context):
        formatted_prompt = f'You are an assistant for answering questions. Use the provided context to generate accurate and concise responses. If the context does not contain sufficient information, respond with "I do not know." Do not fabricate answers.\n\n### Question: {translator_to_en(question)[0]["translation_text"]}\n\n### Context: {context}\n\n### Answer:'
        print(formatted_prompt)
        response = ollama.chat(model='hf.co/lmstudio-community/Llama-3.3-70B-Instruct-GGUF:Q4_K_M', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']
    
    # 4. RAGのセットアップ
    retriever = vectorstore.as_retriever()

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)

    st.success(f"{webpage_url}を正常にロードしました！")
    
    # Webページについて質問する
    prompt = st.text_input("Webページについての質問を入力してください")

    # Webページとチャットする
    if prompt:
        result = rag_chain(prompt)
        st.write(result)
        st.write(translator_to_ja(result)[0]["translation_text"])