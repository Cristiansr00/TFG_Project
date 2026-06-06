import os
import cv2
import numpy as np

KERNELS = {
    1: {"name": "2x2", "value": np.ones((2, 2), np.uint8)},
    2: {"name": "3x3", "value": np.ones((3, 3), np.uint8)},
    3: {"name": "4x4", "value": np.ones((4, 4), np.uint8)},
    4: {"name": "5x5", "value": np.ones((5, 5), np.uint8)}
}

MORPHS = {                                    # Kernel morphological filter
    1: {"name": "Cierre","value": cv2.MORPH_CLOSE},
    2: {"name": "Apertura","value": cv2.MORPH_OPEN},
}

ROJO = "\033[31m"
VERDE = "\033[32m"
AMARILLO = "\033[33m"
AZUL = "\033[34m"
RESET = "\033[0m"
MENU_WIDTH = 50


def color(texto, c):
    return f"{c}{texto}{RESET}"


def limpiar_consola():
    os.system("cls" if os.name == "nt" else "clear")


def mostrar_titulo(titulo):
    limpiar_consola()
    linea = "=" * MENU_WIDTH
    print(f"\n{linea}")
    print(titulo.center(MENU_WIDTH))
    print(f"{linea}\n")


def mostrar_seccion(titulo):
    print(f"\n{titulo}")
    print("-" * len(titulo))


def mostrar_opciones(opciones):
    for clave, texto in opciones:
        print(f"  {clave}. {texto}")


def leer_opcion():
    return input("\nSelecciona una opción: ").strip()


def pausar(mensaje="Pulsa ENTER para continuar..."):
    input(f"\n{mensaje}")


def mostrar_error(mensaje):
    print(color(f"\n[ERROR] {mensaje}", ROJO))


def mostrar_exito(mensaje):
    print(color(f"\n{mensaje}", VERDE))


def listar(lista):
    for index, elemento in enumerate(lista, start=1):
        print(f"  {index}. {elemento}")


def elegir_opcion(opciones, prompt):
    """
    Permite elegir una opción por número o por nombre.
    Devuelve el valor elegido de la lista original.
    """
    opciones = list(opciones)
    if not opciones:
        raise ValueError("No hay opciones disponibles")

    valor = input(prompt).strip()
    if valor.isdigit():
        indice = int(valor) - 1
        if 0 <= indice < len(opciones):
            return opciones[indice]

    if valor in opciones:
        return valor

    raise ValueError(f"Opción no válida: {valor}")


# def yolo_model_exists(model_path: str, model_name: str):
#     if not os.path.exists(model_path):
#         print(f"Modelo {model_path} no encontrado. Descargando...")
#         try:
#             model = YOLO(f"{model_name}.pt")
#             model.save(model_path)
#             print(f"Modelo descargado exitosamente.")
#         except Exception as e:
#             print(f"No se pudo descargar el modelo: {e}")
