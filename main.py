import funciones, constantes


if __name__ == '__main__':
    opciones = {
        'A': funciones.use_gemini,
        'B': funciones.use_llama,
        'C': funciones.use_gemini_with_doc
    }

    index = 1

    print('==============================')

    while True:
        print(f'\n{index}. Seleccione un modelo:')
        print('A) Gemini | Free | Google.')
        print('B) Llama  | Free | HuggingFace.')
        print('C) Gemini | Document | Google.')

        opcion = input('\nOpcion seleccionada: ').strip().upper()

        if opcion == constantes.OPCION_SALIR_VALUE:
            print('\n==============================')
            break

        accion = opciones.get(opcion)

        if not accion:
            print('\nIngrese una opcion valida.')
            continue

        print('\nConfigure los parametros:')
        parametros = {
            'temperature': float(input('Temperature (0.0 - 2.0): ')),
            'top_p': float(input('Top-p (0.0 - 1.0): '))
        }

        if opcion in ('A', 'C'):
            parametros.update({
                'top_k': int(input('Top-k (1 - 100): ')),
                'max_output_tokens': int(input('Max output tokens (0 - 8192): '))
            })

        save_history = input('\n¿Desea mantener el historial de la conversacion? (Y - N): ').strip().upper() == 'Y'

        accion(parametros, save_history)
        
        print('\n==============================')

        index += 1
