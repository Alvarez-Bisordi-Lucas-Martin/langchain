import os, constantes, tokenizador, utils

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS

from huggingface_hub import InferenceClient

from dotenv import load_dotenv


load_dotenv()

GOOGLE_LLM_MODEL = os.getenv('GOOGLE_LLM_MODEL')
GOOGLE_EMBEDDINGS_MODEL = os.getenv('GOOGLE_EMBEDDINGS_MODEL')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

HUGGING_FACE_LLAMA_LLM_MODEL = os.getenv('HUGGING_FACE_LLAMA_LLM_MODEL')
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

DATABASE_URL = os.getenv('DATABASE_URL')

COLLECTION_NAME = os.getenv('COLLECTION_NAME')


def use_gemini(parametros, save_history):
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
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
    document_name = os.path.basename(document_path)
    
    parametros_spliter = None

    if input('\n¿Desea splitear el documento? (Y - N): ').strip().upper() == 'Y':
        parametros_spliter = utils.configurar_parametros_spliter()

        tokenizador_object = tokenizador.TokenizadorRecursiveCharacterTextSplitter()
        document_chunks = tokenizador_object.get_document_chunks(document_path, **parametros_spliter)
    else:
        tokenizador_object = tokenizador.TokenizadorPyPDFLoader()
        document_chunks = tokenizador_object.get_document_chunks(document_path)
    
    document_chunks = tokenizador_object.clean_document_chunks(document_chunks)
    document_chunks = tokenizador_object.set_document_chunks_metadatas(document_chunks)

    document_chunks_ids = tokenizador_object.get_document_chunks_ids(document_chunks, document_name)

    embeddings = tokenizador_object.get_embeddings(GoogleGenerativeAIEmbeddings, GOOGLE_EMBEDDINGS_MODEL, GOOGLE_API_KEY)

    if input('\n¿Desea persistir los embeddings? (Y - N): ').strip().upper() == 'Y':
        vector_store = tokenizador_object.get_or_create_collection(
            embeddings,
            DATABASE_URL,
            COLLECTION_NAME,
            {
                'embeddings_model': GOOGLE_EMBEDDINGS_MODEL,
                'parametros_spliter': parametros_spliter,
                'parametros_retriever': constantes.PARAMETROS_RETRIEVER_DEFAULT,
                'embeddings_count': len(document_chunks)
            }
        )

        tokenizador_object.add_document_chunks(vector_store, document_chunks, document_chunks_ids)
    else:
        FAISS.from_documents(document_chunks, embeddings)
    
    retriever = vector_store.as_retriever(**constantes.PARAMETROS_RETRIEVER_DEFAULT)

    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
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

        print(f'\nRespuesta de Gemini:\n{respuesta.get('answer')}')

        index += 1

        context = respuesta.get('context')

        if not context:
            print(f'\nNo se encontraron embeddings con una similitud mayor o igual a {constantes.PARAMETROS_RETRIEVER_DEFAULT['search_kwargs']['score_threshold']}.')
            continue
        
        print('\nEmbeddings consultados:')
        for index, document in enumerate(context):
            metadata = document.metadata
            
            page = metadata.get('page')
            source = metadata.get('source')
            start_index = metadata.get('start_index')

            print(f'{index}.\nPage: {page}\nSource: {source}\nStart index: {start_index}')


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
            model=HUGGING_FACE_LLAMA_LLM_MODEL,
            messages=messages_history if save_history else messages,
            **parametros
        )

        content = respuesta.choices[0].message.content
        
        print(f'\nRespuesta de Llama:\n{content}')
        
        messages_history.append({'role': 'assistant', 'content': content})

        index += 1
