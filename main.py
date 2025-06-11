import funciones, utils, constantes


if __name__ == '__main__':
    opciones = {
        'A': funciones.use_gemini,
        'B': funciones.use_llama,
        'C': funciones.use_gemini_with_doc
    }

    index = 1

    print('==============================')

    while True:
        opcion = utils.elegir_opcion(index)

        if opcion == constantes.OPCION_SALIR_VALUE:
            print('\n==============================')
            break

        accion = opciones.get(opcion)

        if not accion:
            print('\nValor invalido. Intente nuevamente.')
            continue

        parametros = utils.configurar_parametros(opcion)

        save_history = input('\n¿Desea mantener el historial de la conversacion? (Y - N): ').strip().upper() == 'Y'

        accion(parametros, save_history)
        
        print('\n==============================')

        index += 1
