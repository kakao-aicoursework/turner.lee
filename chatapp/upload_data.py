# base data를 청킹 및 vectorDB로 저장

import os
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = ""

DATA_DIR = os.path.dirname("../doc/")
CHROMA_PERSIST_DIR = os.path.dirname("../db/")
CHROMA_COLLECTION_NAME = "kakao-bot"
KAKAO_SYNC_DATA_FILE_NAME = "project_data_카카오소셜.txt"
KAKAO_SOCIAL_DATA_FILE_NAME = "project_data_카카오싱크.txt"
KAKAOTALK_CHANNEL_DATA_FILE_NAME = "project_data_카카오톡채널.txt"


def upload_embedding_from_file(file_path, meta):
    documents = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    for doc in docs:
        doc.metadata = meta

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


upload_embedding_from_file(os.path.join(DATA_DIR, KAKAO_SYNC_DATA_FILE_NAME), {"service": "KAKAO_SYNC"})
upload_embedding_from_file(os.path.join(DATA_DIR, KAKAO_SOCIAL_DATA_FILE_NAME), {"service": "KAKAO_SOCIAL"})
upload_embedding_from_file(os.path.join(DATA_DIR, KAKAOTALK_CHANNEL_DATA_FILE_NAME), {"service": "KAKAOTALK_CHANNEL"})


from pprint import pprint

db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)

# 데이터 입력 확인
docs = db.get(where={"service": "KAKAO_SYNC"}, limit=3)
pprint(docs)

docs = db.get(where={"service": "KAKAO_SOCIAL"}, limit=3)
pprint(docs)

docs = db.get(where={"service": "KAKAOTALK_CHANNEL"}, limit=3)
pprint(docs)
