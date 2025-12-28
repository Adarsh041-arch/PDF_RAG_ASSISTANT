from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from langchain_core.prompts import PromptTemplate
# from langchain.schema import Document

load_dotenv()

def build_rag_chain(pdf_path:str):
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser=StrOutputParser()

    loader=PyPDFLoader(pdf_path)
    docs=loader.load()
    Splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks=Splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    Vdatabase = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    retriver=Vdatabase.as_retriever(
        search_type='mmr',
        search_kwargs={'k':8,'lambda_mult':0.3},
    )
    prompt1=PromptTemplate(
        template="""Classify the user question.

    If the question is casual conversation, greeting, or chit-chat
    (e.g. hi, hello, how are you), respond with ONLY: CHAT

    If the question requires understanding the document as a whole
    (e.g. duration, total weeks, overall summary),
    respond with ONLY: GLOBAL

    If it can be answered from a specific section or topic,
    respond with ONLY: LOCAL

    Respond with ONLY ONE WORD.

    Question:
    {question}"""
    ,
        input_variables=["question"]
    )

    prompt2=PromptTemplate(
    template= """You are a helpful assistant.
    Answer the question strictly based on the context below.

    Context:
    {context}

    Question:
    {question}
    """,
    input_variables=["context","question"]
    )

    prompt3=PromptTemplate(
        template="""
    You are given the complete document content below.
    Answer the question accurately.

    Document:
    {document}

    Question:
    {question}

    Answer:
    """,
        input_variables=["document", "question"]
    )

    # Document → string formatter
    format_docs = RunnableLambda(
        lambda docs: "\n\n".join(doc.page_content for doc in docs)
    )
    full_document_text = "\n\n".join(doc.page_content for doc in docs)
    parallel_chain=RunnableParallel({
        'context':retriver | format_docs,
        'question':RunnablePassthrough()
    })
    parallel_chain_global=RunnableParallel({
        'document':RunnableLambda(lambda _: full_document_text),
        'question':RunnablePassthrough()
    })

    # Chains
    chain_chat=model|parser
    chain_local=parallel_chain | prompt2 | model |parser
    chain_global=parallel_chain_global | prompt3 |model |parser
    classifier_chain=prompt1 | model| parser

    #Routes

    route_to_chat = RunnableLambda(lambda x: x["question"]) | chain_chat
    route_to_local = RunnableLambda(lambda x: x["question"]) | chain_local
    route_to_global = RunnableLambda(lambda x: x["question"]) | chain_global


    final_chain = (
        RunnableParallel(
            {
                "route": classifier_chain,
                "question": RunnablePassthrough(),
            }
        )
        | RunnableBranch(
            (lambda x: x["route"].strip() == "LOCAL", route_to_local),
            (lambda x: x["route"].strip() == "GLOBAL",route_to_global),
            (lambda x: x["route"].strip() == "CHAT",route_to_chat),
            route_to_local,

        )
    )
    return final_chain


