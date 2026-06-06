from .menu_modelos import menu_modelos
from .menu_metricas import menu_metricas
from .menu_datasets import menu_datasets
from utils.tools import limpiar_consola, color, ROJO


def menu_principal():
    while True:
        menu_principal_options()
        opcion = input("\nSelecciona una opcion: ").strip()

        if opcion == "1":
            menu_modelos()
        elif opcion == "2":
            menu_datasets()
        elif opcion == "3":
            menu_metricas()
        elif opcion == "0":
            print("\nSaliendo del programa")
            break
        else:
            print(color("\n[ERROR] Opcion no valida. Por favor, selecciona una opcion valida.", ROJO))
            input("Presiona Enter para continuar...")


def menu_principal_options():
    limpiar_consola()
    print("\n" + "=" * 40)
    print(" " * 13 + "MENU PRINCIPAL")
    print("=" * 40 + "\n")
    print("1. Gestion de modelos")
    print("2. Gestion de datasets")
    print("3. Metricas y evaluacion")
    print("0. Salir")
