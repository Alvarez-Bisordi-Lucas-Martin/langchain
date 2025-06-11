import constantes


def elegir_opcion(index):
    print(f'\n{index}. Seleccione un modelo:')
    print('A) Gemini | Free | Google.')
    print('B) Llama  | Free | HuggingFace.')
    print('C) Gemini | Document | Google.')

    return input('\nOpcion seleccionada: ').strip().upper()


def configurar_parametros(opcion):
    if input('\n¿Desea configurar los parametros? (Y - N): ').strip().upper() != 'Y':
        if opcion in ('A', 'C'):
            return constantes.PARAMETROS_DEFAULT
        
        return {
            'temperature': constantes.PARAMETROS_DEFAULT['temperature'],
            'top_p': constantes.PARAMETROS_DEFAULT['top_p']
        }
    
    parametros = {
        'temperature': pedir_valor('Temperature', 0.0, 2.0, float),
        'top_p': pedir_valor('Top-p', 0.0, 1.0, float)
    }

    if opcion in ('A', 'C'):
        parametros['top_k'] = pedir_valor('Top-k', 1, 100, int)
        parametros['max_output_tokens'] = pedir_valor('Max output tokens', 0, 8192, int)

    return parametros


def pedir_valor(mensaje, minimo, maximo, tipo):
    while True:
        try:
            valor = tipo(input(f'{mensaje} ({minimo} - {maximo}): '))

            if minimo < valor < maximo:
                return valor
        except ValueError:
            pass

        print('Valor inválido. Intente nuevamente.')
