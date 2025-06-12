OPCION_SALIR_VALUE = '@SALIR'

CONTEXT_VALUE = 'Eres un asistente personal.'
CONTEXT_WITH_DOC_VALUE = CONTEXT_VALUE + '\n\nResponde a partir de esta informacion:\n\n{context}'
CONTEXT_WITH_DOC_AND_HISTORY_VALUE = CONTEXT_WITH_DOC_VALUE + '\n\nDebes responder unicamente la ultima pregunta del usuario.'

COLLECTION_NAME = 'POC-LANGCHAIN-V0.0.1'

PARAMETROS_LLM_DEFAULT = {
    'temperature': 0.2,
    'top_p': 0.85,
    'top_k': 30,
    'max_output_tokens': 400
}

PARAMETROS_SPLITER_DEFAULT = {
    'chunck_size': 2000,
    'chunck_overlap': 0,
    'length_function': len,
    'separators': ['\n\n', '\n', ' ', ''],
    'add_start_index': False,
    'keep_separator': False,
    'is_separator_regex': False
}
