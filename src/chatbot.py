from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI  # გამოიყენე Google Gemini API LLM adapter თუ გინდა
from langchain.vectorstores import FAISS

def create_chatbot(vectorstore: FAISS):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = ChatOpenAI(temperature=0)  # LLM აქ შეიძლება შეიცვალოს Google Gemini API adapter-ზე
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

def ask_question(chatbot, question: str):
    response = chatbot({"question": question})
    answer = response['answer']
    sources = [doc.metadata.get("source", "unknown") for doc in response['source_documents']]
    return answer, sources
