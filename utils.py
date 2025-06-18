import constantes


def elegir_opcion(index):
    print(f'\n{index}. Seleccione un modelo:')
    print('A) Gemini | Free | Google.')
    print('B) Llama  | Free | HuggingFace.')
    print('C) Gemini | Document | Google.')

    return input('\nOpcion seleccionada: ').strip().upper()


def configurar_parametros_llm(opcion):
    if input('\n¿Desea configurar los parametros del llm? (Y - N): ').strip().upper() != 'Y':
        if opcion in ('A', 'C'):
            return constantes.PARAMETROS_LLM_DEFAULT.copy()
        
        return {
            'temperature': constantes.PARAMETROS_LLM_DEFAULT['temperature'],
            'top_p': constantes.PARAMETROS_LLM_DEFAULT['top_p']
        }
    
    parametros = {
        'temperature': pedir_valor('Temperature', 0.1, 1.9, float),
        'top_p': pedir_valor('Top-p', 0.1, 0.9, float)
    }

    if opcion in ('A', 'C'):
        parametros['top_k'] = pedir_valor('Top-k', 1, 100, int)
        parametros['max_output_tokens'] = pedir_valor('Max output tokens', 1, 8000, int)

    return parametros


def configurar_parametros_splitter():
    if input('\n¿Desea configurar los parametros del splitter? (Y - N): ').strip().upper() != 'Y':
        return constantes.PARAMETROS_SPLITTER_DEFAULT.copy()

    return {
        'chunk_size': pedir_valor('Chunk size', 1000, 4000, int),
        'chunk_overlap': pedir_valor('Chunk overlap', 0, 200, int),
        'length_function': constantes.PARAMETROS_SPLITTER_DEFAULT['length_function'],
        'separators': constantes.PARAMETROS_SPLITTER_DEFAULT['separators'],
        'add_start_index': pedir_valor('Add start index', 0, 1, bool),
        'keep_separator': pedir_valor('Keep separator', 0, 1, bool),
        'is_separator_regex': pedir_valor('Is separator regex', 0, 1, bool)
    }


def pedir_valor(mensaje, minimo, maximo, tipo):
    while True:
        try:
            valor = tipo(input(f'{mensaje} ({minimo} - {maximo}): '))

            if minimo <= valor <= maximo:
                return valor
        except ValueError:
            print('\nValor invalido. Intente nuevamente.\n')
