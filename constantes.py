OPCION_SALIR_VALUE = '@SALIR'

CONTEXT_VALUE = 'Eres un asistente personal.'
CONTEXT_WITH_DOC_VALUE = CONTEXT_VALUE + '\n\nResponde a partir de esta informacion:\n\n{context}'
CONTEXT_WITH_DOC_AND_HISTORY_VALUE = CONTEXT_WITH_DOC_VALUE + '\n\nDebes responder unicamente la ultima pregunta del usuario.'

PARAMETROS_LLM_DEFAULT = {
    'temperature': 0.2,
    'top_p': 0.85,
    'top_k': 30,
    'max_output_tokens': 500
}

PARAMETROS_SPLITTER_DEFAULT = {
    'chunk_size': 3000,
    'chunk_overlap': 0,
    'length_function': len,
    'separators': ['.', '\n'],
    'add_start_index': True,
    'keep_separator': False,
    'is_separator_regex': False
}

PARAMETROS_RETRIEVER_DEFAULT = {
    'search_type': 'similarity_score_threshold',
    'search_kwargs': {
        'k': 6,
        'score_threshold': 0.6
    }
}
