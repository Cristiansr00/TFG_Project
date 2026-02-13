from services.models_service import get_trained_models
from services.tests_service import testAndsave
from services.dataset_service import get_datasets
from utils.tools import limpiar_consola


def menu_metricas():
    while True:
        limpiar_consola()
        menu_metricas_options()
        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            # generar_metricas()
            print('generar_metricas')
            print(f'{get_trained_models()}')
            model = input("Introduce el nombre del modelo:")

            print(f'{get_datasets()}')
            dataset = input("Introduce el nombre del dataset:")

            folds = input("Introduce el numero de folds:")
            testAndsave("",model, dataset, int(folds))
        elif opcion == "2":
            # comparar_modelos()
            print('comparar_modelos')
        elif opcion == "0":
            break
        else:
            print("[ERROR] Opción no válida")

def menu_metricas_options():
    print("\n--- MENÚ MÉTRICAS ---")
    print("1. Evaluar modelo")
    print("2. Comparar modelos")
    print("0. Volver")
