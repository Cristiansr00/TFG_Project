from .menu_modelos import menu_modelos
from .menu_metricas import menu_metricas
from .menu_datasets import menu_datasets
from .menu_manual import menu_manual
from utils.tools import mostrar_error, mostrar_opciones, mostrar_titulo, leer_opcion, pausar


def menu_principal():
    while True:
        menu_principal_options()
        opcion = leer_opcion()

        if opcion == "1":
            menu_modelos()
        elif opcion == "2":
            menu_datasets()
        elif opcion == "3":
            menu_metricas()
        elif opcion == "4":
            menu_manual()
        elif opcion == "0":
            print("\nSaliendo del programa")
            break
        else:
            mostrar_error("Opción no válida. Por favor, selecciona una opción válida.")
            pausar()


def menu_principal_options():
    mostrar_titulo("MENÚ PRINCIPAL")
    mostrar_opciones([
        ("1", "Gestión de modelos"),
        ("2", "Gestión de datasets"),
        ("3", "Métricas y evaluación"),
        ("4", "Manual de uso"),
        ("0", "Salir"),
    ])
