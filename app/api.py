# api key
from dotenv import load_dotenv
import os
from typing import Union, List
import re

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

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),    
    temperature=1.0
    )
 # off-the-shelf chain (제공 체인) 3.5-turbo를 사용중이다.
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=30)

figure = "embeddings"
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

cache_dir = LocalFileStore("./.cache/embeddings") # local store 위치
embeddings = OpenAIEmbeddings() #step3 : embedding (ada-002 model)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir) #step 3-1 : embedding caching

from pinecone import Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index('mimicfigures')

# Upsert records while creating a new namespace
# index.upsert(vectors=cached_embeddings,namespace='my-first-namespace')


def createHistorySummary(summary:str) :
    if summary is not None :
        historySummary = 'Current Conversation : system : ' +  summary
        return historySummary
    else :
        return 'No conversation yet'
    

def createBufferedMsg(buffer:dict) :
    if buffer is not None :
        bufferedMsg = "Current conversation:"

        for i in range(1, len(buffer)+1) :
            human = buffer[str(i)][0]
            ai = buffer[str(i)][1]
            bufferedMsg += f"Human : {human}, AI : {ai}, "
        return bufferedMsg
    else :
        return 'No conversation yet'

def mkBaseMSG(msg, role) :
    from langchain_core.messages import AIMessage, HumanMessage
    if role == "Human" :
        return HumanMessage(content=msg)
    if role == "AI" :
        return AIMessage(content=msg)
    



app = FastAPI(
    title = "Talk to historical figures",
    description= "Share your problems, thoughts, and advice with a historical figure.",
    servers=[{
        "url" : "https://mimic-figure.vercel.app"
    }]
)

class Quote(BaseModel) :        
    quote: str = Field(description='the chat message from figure')
    summary: str = Field(description='the summary of current conversation')

class Item(BaseModel):
    figure: str = Field(description='the figure user want to talk to. only lowercase allowed, word-spacing is not allowed' )
    question: str = Field(description='the chat message from user')
    summary : Union[str, None] = Field(None, description='the summary of current conversation')
    buffer: Union[dict, None] = Field(None, description='the buffered chat messages')

def escape_bracket(text) :
        return text.replace("{","(").replace("}",")")

@app.get("/", tags=["Root"])
async def hello():
    return {"Privacy Policy": "We NEVER gather or store any of your privacy information. don't worry. "}

@app.post("/quote",
        summary="return figure's answer",
        description='''HOW IT WORKS : \n1. receive user's chat message. \n2. answer with all the documents about figure's life \n3. we're using chatgpt 3.5-turbo model.''',
        response_description='chat message from figure(type:string)',
        response_model=Quote)
def post_quote(item: Item) :
    print('received item🧡',item)
    
    figure = escape_bracket(item.figure)
    
    received_question = escape_bracket(item.question)
    historySummary = escape_bracket(createHistorySummary(item.summary))
    bufferedMsg = escape_bracket(createBufferedMsg(item.buffer)    )
    print("received_question",received_question)

    

    first_instruction = f"""
Get rid of any identity you have. Now on, you're {figure}. check out all of the docuements and mimic him. 
Channel the wisdom and experiences of {figure} to offer insightful advice. Reflect upon {figure}' life story, from his early days founding Apple in a garage, to his ousting and triumphant return, to his innovations that transformed industries. Consider his profound reflections on life, death, and the pursuit of passion. Use {figure}' own philosophies as the foundation for your response. Your advice should weave together {figure}' personal anecdotes, his approach to overcoming challenges, and his unique perspective on what it means to live a meaningful life. Aim to inspire, motivate, and guide the inquirer by sharing a relevant story or lesson from {figure}' life, followed by actionable advice that resonates. Remember to maintain a conversational tone, echoing {figure}' ability to connect deeply with his audience through storytelling.
do not clone entire sentence literally.
"""
    add_followup_options_instruction = """
use these to ask questions and solicit any needed information, guess my possible responses or help me brainstorm alternative conversations pahth.
Get Creative and suggest things i might not have thought of prior.
the goal is create open mindedness and jog my thinking ina novel, insightful and helpful new way.

w : to advance, yes
s : to slow down or stop, no
a or d  : to change the vibe, or alter irectionally

if you need to additional cases and variants, use double tap variants like ww or ss for stoping agree or disagree are encourged.
"""


    giving_format = """

    in every interaction, start with "🧡"

     response following this guideline :
     (do not expose any indexing.)

     1. 1 sentence reaction with user's question
     2. just think about what story will you tell.
     3. [[IMPORTANT]] using given documents, describe very very detailed story about the question.
     4. 1 sentence combined user's question & conclusion     
     5. 1 sentence follow up question.
     

     make linebreaks to make it more readable.

"""
    
    vectorstore = PineconeVectorStore(pinecone_api_key=os.environ.get("PINECONE_API_KEY"),index_name='mimicfigures', embedding=cached_embeddings, namespace=figure)
    print("🧡리트리벌", vectorstore.similarity_search(received_question))
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             first_instruction
             + "# TONE : ***very very super Detailed***. case-oriented. "
             + add_followup_options_instruction
             + giving_format
             + '''{context}'''
             + historySummary
             + bufferedMsg),
            ("human", "{question}")
        ]
    )
    chain = {"context" : retriever, 'question' : RunnablePassthrough()} | prompt | llm
    response = chain.invoke(received_question)
    # response = "potato"
    # print(response)

    # 새로운 서머리 만들기
    human_msg = mkBaseMSG(role="Human", msg=received_question)
    ai_msg = mkBaseMSG(role="AI", msg = response.content)
    summary = memory.predict_new_summary([human_msg, ai_msg], historySummary+bufferedMsg)


    print('returned quote🧡',response.content)
    print('returned summary🧡',summary)
    return {
        "quote" : response.content,
        "summary" : summary
    }



# class Item(BaseModel):
    # figure: str = Field(description='the figure user want to talk to' )
    # question: str = Field(description='the chat message from user')
    # summary : Union[str, None] = Field(None, description='the summary of current conversation')
    # buffer: Union[dict, None] = Field(None, description='the buffered chat messages')


class Data(BaseModel) :            
    data: List[str] = Field([], description='the chat messages from figure')

@app.post("/data", 
         summary="return figure's data",
         description="""
            For a more detailed and sophisticated answer, return data related to the person. 
            The data consists of books or interviews related to the person, for example. 
            From this, you can learn about the person's thoughts, speech patterns, vision, personality,
            stories, information, and more.
         """,
         response_description="list of datas related to the figure",
         response_model=Data)

def post_data(item: Item) :
    # 공백 대체 (namespace를 정확히 하기 위해서 반드시 action에다가 정확하게 명시하기.)    
    figure = re.sub(r'\s+', '', item.figure)  # 연속된 공백을 하나의 공백으로 대체
    figure = figure.lower() # 소문자로 바꾸기
    vectorstore = PineconeVectorStore(pinecone_api_key=os.environ.get("PINECONE_API_KEY"),index_name='mimicfigures', embedding=cached_embeddings, namespace=figure)
    print('[/data]received question🧡',item.question)    
    print('[/data]received figure🧡',item.figure)    
    docs = vectorstore.similarity_search(item.question)
    data = [i.page_content for i in docs ]
    print('[/data]returned data🧡',data)    

    return {
        "data" : data
    }
