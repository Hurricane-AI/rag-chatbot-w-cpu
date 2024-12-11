import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

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
    embeddings = OllamaEmbeddings(model="hf.co/ChristianAzinn/mxbai-embed-large-v1-gguf:Q5_K_M")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # 3. Ollama Llama3モデルを呼び出す
    def ollama_llm(question, context):
        formatted_prompt = f"<s>以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n次の文章の情報を元に、与えられた質問に答えなさい。\n\n### 文章:\n{context}\n\n### 質問:\n{question}\n\n応答: "
        response = ollama.chat(model='hf.co/team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-GGUF:Q5_K_M', messages=[{'role': 'user', 'content': formatted_prompt}])
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