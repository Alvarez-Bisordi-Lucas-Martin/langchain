import re, hashlib

from abc import ABC, abstractmethod

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_postgres.vectorstores import PGVector


class Tokenizador(ABC):
    @abstractmethod
    def get_document_chunks(self):
        pass
    
    def clean_document_chunks(self, document_chunks):
        for document_chunk in document_chunks:
            page_content = document_chunk.page_content

            # Eliminar caracteres nulos
            page_content = page_content.replace('\x00', '')
            # Eliminar espacios externos
            page_content = page_content.strip()
            # Eliminar espacios internos duplicados
            page_content = re.sub(r' {2,}', ' ', page_content)
            # Eliminar saltos de linea internos con espacios
            page_content = re.sub(r' *\n *', '\n', page_content)

            document_chunk.page_content = page_content
        
        return document_chunks
    
    def set_document_chunks_metadatas(self, document_chunks):
        for document_chunk in document_chunks:
            document_chunk.metadata = {
                key: document_chunk.metadata.get(key)
                for key in ('page', 'source', 'start_index')
            }
        
        return document_chunks
    
    def get_document_chunks_ids(self, document_chunks, document_name):
        return [
            f'{document_name}_{hashlib.sha256(document_chunk.page_content.encode('utf-8')).hexdigest()}'
            for document_chunk in document_chunks
        ]
    
    def get_embeddings(self, embeddings_class, embeddings_model, api_key):
        return embeddings_class(
            model=embeddings_model,
            google_api_key=api_key
        )
    
    def get_or_create_collection(self, embeddings, database_url, collection_name, collection_metadata=None, use_jsonb=True):
        return PGVector(
            embeddings=embeddings,
            connection=database_url,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            use_jsonb=use_jsonb
        )
    
    def add_document_chunks(self, vector_store, document_chunks, document_chunks_ids=None):
        vector_store.add_documents(document_chunks, ids=document_chunks_ids)


class TokenizadorPyPDFLoader(Tokenizador):
    def get_document_chunks(self, document_path):
        document_loader = PyPDFLoader(document_path)

        return document_loader.load()


class TokenizadorRecursiveCharacterTextSplitter(Tokenizador):
    def get_document_chunks(self, document_path, chunk_size=3000, chunk_overlap=0, length_function=len, separators=['.', '\n'], add_start_index=True, keep_separator=False, is_separator_regex=False):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            add_start_index=add_start_index,
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex
        )

        tokenizador = TokenizadorPyPDFLoader()
        
        document_chunks = tokenizador.get_document_chunks(document_path)
        
        return text_splitter.split_documents(document_chunks)
