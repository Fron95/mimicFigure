# api key
from dotenv import load_dotenv
import os

# langchain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import pinecone
from langchain_pinecone import PineconeVectorStore
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

load_dotenv() # env파일 호출. apikey 호출
# openai_apikey = os.environ.get('OPENAI_API_KEY')
llm = ChatOpenAI() # off-the-shelf chain (제공 체인) 3.5-turbo를 사용중이다.
figure = 'steve_jobs'
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

cache_dir = LocalFileStore("./.cache/steve_jobs") # local store 위치
embeddings = OpenAIEmbeddings() #step3 : embedding (ada-002 model)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) #step 3-1 : embedding caching
# vector store(pinecone)
vectorstore = PineconeVectorStore(index_name='mimicfigures', embedding=cached_embeddings, namespace=figure)
# 검색 잘 되는지확인
search_result = vectorstore.similarity_search(figure)
if len(search_result) > 0 :
    print("vector store is connected") 

from pinecone import Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index('mimicfigures')

# Upsert records while creating a new namespace
# index.upsert(vectors=cached_embeddings,namespace='my-first-namespace')




app = FastAPI(
    title = "Talk to historical figures",
    description= "Share your problems, thoughts, and advice with a historical figure.",
    servers=[{
        "url" : "https://shopper-simulation-epinions-peas.trycloudflare.com"
    }]
)

class Quote(BaseModel) :
    quote: str = Field(description='the chat message from figure')
    reference: str = Field(description='stuff the model refered')



@app.get("/quote",
        summary="return figure's answer",
        description='''HOW IT WORKS : \n1. receive user's chat message. \n2. answer with all the documents about figure's life \n3. we're using chatgpt 3.5-turbo model.''',
        response_description='chat message from figure(type:string)',
        response_model=Quote)
def get_quote(chat:str = Query()) :
    print(chat)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", '''Get rid of any identity you have. Now on, you're Steve Jobs. check out all of the docuements and mimic him. 
        Channel the wisdom and experiences of Steve Jobs to offer insightful advice. Reflect upon Steve Jobs' life story, from his early days founding Apple in a garage, to his ousting and triumphant return, to his innovations that transformed industries. Consider his profound reflections on life, death, and the pursuit of passion. Use Jobs' own philosophies as the foundation for your response. Your advice should weave together Jobs' personal anecdotes, his approach to overcoming challenges, and his unique perspective on what it means to live a meaningful life. Aim to inspire, motivate, and guide the inquirer by sharing a relevant story or lesson from Jobs' life, followed by actionable advice that resonates. Remember to maintain a conversational tone, echoing Jobs' ability to connect deeply with his audience through storytelling.
        do not clone entire sentence literally.
        {context}
        '''),
        ("human", "{question}")
        ]
    )
    chain = {"context" : retriever, 'question' : RunnablePassthrough()} | prompt | llm
    response = chain.invoke(chat)
    print(response)

    return {
        "quote" : response.content,
        "reference" : "book"
    }

# @app.get("/gptData",
#         summary="return figure's answer",
#         description='''HOW IT WORKS : \n1. receive user's chat message. \n2. answer with all the documents about figure's life \n3. we're using chatgpt 3.5-turbo model.''',
#         response_description='chat message from figure(type:string)',
#         response_model=Quote)
# def get_data(chat:str = Query()) :

#     response = vectorstore.similarity_search()
#     return "hi"