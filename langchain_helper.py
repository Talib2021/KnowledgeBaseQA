from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Configure Google Generative AI with your API key
genai.configure(api_key="Your_api_key")

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    """
    Create a FAISS vector database from a CSV file and save it locally.
    """
    loader = CSVLoader(file_path='data_faqs.csv', source_column="prompt")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    """
    Load the FAISS vector database and create a RetrievalQA chain.
    """
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't fabricate an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=genai
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

if __name__ == "__main__":
    create_vector_db()

    chain = get_qa_chain()

    question = "Do you have a JavaScript course?"
    response = chain({"query": question})

    print(response["result"])
