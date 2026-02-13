from .menu_modelos import menu_modelos
from .menu_metricas import menu_metricas
from .menu_datasets import menu_datasets
from utils.tools import limpiar_consola

def menu_principal():
    while True:
        limpiar_consola()
        menu_principal_options()

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            menu_modelos()
        elif opcion == "2":
            menu_datasets()
        elif opcion == "3":
            menu_metricas()
        elif opcion == "0":
            print("\nSaliendo del programa 👋")
            break
        else:
            print("[ERROR] Opción no válida")

def menu_principal_options():
    print("\n" + "="*40)
    print(" MENÚ PRINCIPAL ")
    print("="*40)
    print("1. Gestión de modelos")
    print("2. Gestión de datasets")
    print("3. Métricas y evaluación")
    print("0. Salir")