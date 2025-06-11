import os, constantes, hashlib

from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_postgres.vectorstores import PGVector

from huggingface_hub import InferenceClient


load_dotenv()

GOOGLE_MODEL = os.getenv('GOOGLE_MODEL')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

HUGGING_FACE_LLAMA_MODEL = os.getenv('HUGGING_FACE_LLAMA_MODEL')
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

DATABASE_URL = os.getenv('DATABASE_URL')


def use_gemini(parametros, save_history):
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        **parametros
    )

    messages_history = [
        SystemMessage(content=constantes.CONTEXT_VALUE)
    ]

    index = 1

    while True:
        prompt = input(f'\n{index}. Ingrese un mensaje: ')

        if prompt.upper() == constantes.OPCION_SALIR_VALUE: break
        
        messages = [
            SystemMessage(content=constantes.CONTEXT_VALUE),
            HumanMessage(content=prompt)
        ]

        messages_history.append(HumanMessage(content=prompt))

        respuesta = llm.invoke(messages_history if save_history else messages)

        print(f'\nRespuesta de Gemini:\n{respuesta.content}')

        messages_history.append(AIMessage(content=respuesta.content))

        index += 1


def use_gemini_with_doc(parametros, save_history):
    document_path = r'C:\Users\DELL\OneDrive\Documentos\Ecom\Clientes\Contaduria General\Documentos\v2\DOCUMENTO N°4 - LAF 1092A.pdf'
    document_loader = PyPDFLoader(document_path)
    
    document_chunks = document_loader.load()
    document_chunks_ids = []
    
    for document_chunk in document_chunks:
        content_hash = hashlib.sha256(document_chunk.page_content.encode('utf-8')).hexdigest()

        document_chunk.metadata['id'] = f'{os.path.basename(document_path)}_{content_hash}'

        document_chunks_ids.append(document_chunk.metadata['id'])
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=GOOGLE_API_KEY
    )

    persistir = input('\n¿Desea persistir los embeddings? (Y - N): ').strip().upper() == 'Y'

    vector_store = (
        PGVector.from_documents(
            documents=document_chunks,
            embedding=embeddings,
            connection=DATABASE_URL,
            collection_name=constantes.COLLECTION_NAME,
            use_jsonb=True,
            ids=document_chunks_ids
        )
        if persistir
        else FAISS.from_documents(document_chunks, embeddings)
    )
    
    retriever = vector_store.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        **parametros
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ('system', constantes.CONTEXT_WITH_DOC_AND_HISTORY_VALUE if save_history else constantes.CONTEXT_WITH_DOC_VALUE),
        ('human', '{input}')
    ])
    rag_context = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, rag_context)

    human_messages_history = []

    index = 1

    while True:
        prompt = input(f'\n{index}. Ingrese un mensaje: ')

        if prompt.upper() == constantes.OPCION_SALIR_VALUE: break

        human_messages_history.append(prompt)

        respuesta = rag_chain.invoke(
            {'input': ' - '.join(message for message in human_messages_history) if save_history else prompt}
        )

        print(f'\nRespuesta de Gemini:\n{respuesta['answer']}')

        index += 1


def use_llama(parametros, save_history):
    llm = InferenceClient(token=HUGGING_FACE_API_KEY)
    
    messages_history = [
        {'role': 'system', 'content': constantes.CONTEXT_VALUE}
    ]

    index = 1

    while True:
        prompt = input(f'\n{index}. Ingrese un mensaje: ')

        if prompt.upper() == constantes.OPCION_SALIR_VALUE: break

        messages = [
            {'role': 'system', 'content': constantes.CONTEXT_VALUE},
            {'role': 'user', 'content': prompt}
        ]

        messages_history.append({'role': 'user', 'content': prompt})

        respuesta = llm.chat_completion(
            model=HUGGING_FACE_LLAMA_MODEL,
            messages=messages_history if save_history else messages,
            **parametros
        )

        content = respuesta.choices[0].message.content
        
        print(f'\nRespuesta de Llama:\n{content}')
        
        messages_history.append({'role': 'assistant', 'content': content})

        index += 1
