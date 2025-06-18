import re, hashlib

from abc import ABC, abstractmethod

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_postgres.vectorstores import PGVector


class Tokenizador(ABC):
    def __init__(self):
        self.document_chunks = None
        self.vector_store = None
    
    @abstractmethod
    def create_document_chunks(self):
        pass
    
    def get_document_chunks(self):
        return self.document_chunks
    
    def reset_document_chunks(self):
        self.document_chunks = None
    
    def clean_document_chunks(self):
        for document_chunk in self.document_chunks:
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
    
    def set_document_chunks_metadatas(self, include_metadatas=None):
        if include_metadatas:
            for document_chunk in self.document_chunks:
                document_chunk.metadata = {
                    key: document_chunk.metadata.get(key)
                    for key in include_metadatas
                }
    
    def get_document_chunks_ids(self, document_name):
        return [
            f'{document_name}_{hashlib.sha256(document_chunk.page_content.encode('utf-8')).hexdigest()}'
            for document_chunk in self.document_chunks
        ]
    
    def create_collection(self, embeddings_constructor, database_url, collection_name, collection_metadata=None, use_jsonb=True):
        self.vector_store = PGVector(
            embeddings=embeddings_constructor,
            connection=database_url,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            use_jsonb=use_jsonb
        )
    
    def get_collection(self):
        return self.vector_store
    
    def add_document_chunks(self, document_chunks_ids=None):
        self.vector_store.add_documents(self.document_chunks, ids=document_chunks_ids)


class TokenizadorPyPDFLoader(Tokenizador):
    def __init__(self, document_path):
        super().__init__()

        self.document_path = document_path
    
    def create_document_chunks(self):
        document_loader = PyPDFLoader(self.document_path)

        self.document_chunks = document_loader.load()


class TokenizadorRecursiveCharacterTextSplitter(Tokenizador):
    def __init__(self, document_path, chunk_size=3000, chunk_overlap=0, length_function=len, separators=['.', '\n'], add_start_index=True, keep_separator=False, is_separator_regex=False):
        super().__init__()

        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators
        self.add_start_index = add_start_index
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
    
    def create_document_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.separators,
            add_start_index=self.add_start_index,
            keep_separator=self.keep_separator,
            is_separator_regex=self.is_separator_regex
        )

        tokenizador_object = TokenizadorPyPDFLoader(self.document_path)

        tokenizador_object.create_document_chunks()

        self.document_chunks = text_splitter.split_documents(tokenizador_object.get_document_chunks())
