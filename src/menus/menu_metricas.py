from services.models_service import get_trained_models
from services.tests_service import testAndsave, compare_consolidated_results
from services.dataset_service import get_datasets
from utils.tools import (
    elegir_opcion,
    leer_opcion,
    listar,
    mostrar_error,
    mostrar_opciones,
    mostrar_seccion,
    mostrar_titulo,
    pausar,
)


def menu_metricas():
    while True:
        menu_metricas_options()
        opcion = leer_opcion()

        if opcion == "1":
            try:
                # Evalúa un modelo entrenado y actualiza CSV/Excel de resultados.
                evaluar_modelo()
            except (ValueError, FileNotFoundError) as e:
                mostrar_error(f"Error al evaluar modelo: {e}")
            pausar("Pulsa ENTER para volver al menú...")
        elif opcion == "2":
            try:
                # Lee el Excel consolidado y muestra un ranking sencillo por F1.
                comparar_modelos()
            except FileNotFoundError:
                print("\nTodavía no hay resultados consolidados. Evalúa algún modelo primero.")
            except ValueError as e:
                mostrar_error(f"Error al comparar modelos: {e}")
            pausar("Pulsa ENTER para volver al menú...")
        elif opcion == "0":
            break
        else:
            mostrar_error("Opción no válida.")
            pausar()


def menu_metricas_options():
    mostrar_titulo("MENÚ MÉTRICAS")
    mostrar_opciones([
        ("1", "Evaluar modelo"),
        ("2", "Comparar modelos"),
        ("0", "Volver"),
    ])


def evaluar_modelo():
    modelos = get_trained_models()
    datasets = get_datasets()
    if not modelos:
        print("No hay modelos entrenados en 'models/trained'.")
        return
    if not datasets:
        print("No hay datasets en 'data/processed'.")
        return

    mostrar_seccion("Modelos entrenados")
    listar(modelos)
    model = elegir_opcion(modelos, "Introduce el número o nombre del modelo: ")

    mostrar_seccion("Datasets disponibles")
    listar(datasets)
    dataset = elegir_opcion(datasets, "Introduce el número o nombre del dataset: ")

    folds = int(input("Introduce el número de folds: ").strip())
    configuration = input("Nombre de configuración para guardar resultados [modelo]: ").strip()
    testAndsave(configuration, model, dataset, folds)


def comparar_modelos():
    metric_labels = {"Detection": "Detección", "Classification": "Clasificación"}
    for metric_type, label in metric_labels.items():
        mostrar_seccion(f"Top F1 - {label}")
        resultados = compare_consolidated_results(metric="F1-Score", metric_type=metric_type, limit=10)
        if not resultados:
            print("Sin resultados disponibles")
            continue
        for index, row in enumerate(resultados, start=1):
            print(f"{index}. {row['configuration']} -> {row['value']:.4f}")
