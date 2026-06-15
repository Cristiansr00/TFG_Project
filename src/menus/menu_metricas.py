from services.dataset_service import get_datasets
from services.models_service import get_trained_models
from services.tests_service import (
    compare_consolidated_results,
    get_model_weight_files,
    run_image_inference,
    testAndsave,
)
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
                evaluar_modelo()
            except (ValueError, FileNotFoundError) as e:
                mostrar_error(f"Error al evaluar modelo: {e}")
            pausar("Pulsa ENTER para volver al menu...")
        elif opcion == "2":
            try:
                comparar_modelos()
            except FileNotFoundError:
                print("\nTodavia no hay resultados consolidados. Evalua algun modelo primero.")
            except ValueError as e:
                mostrar_error(f"Error al comparar modelos: {e}")
            pausar("Pulsa ENTER para volver al menu...")
        elif opcion == "3":
            try:
                inferencia_imagen()
            except (ValueError, FileNotFoundError, OSError) as e:
                mostrar_error(f"Error al aplicar inferencia: {e}")
            pausar("Pulsa ENTER para volver al menu...")
        elif opcion == "0":
            break
        else:
            mostrar_error("Opcion no valida.")
            pausar()


def menu_metricas_options():
    mostrar_titulo("MENU METRICAS")
    mostrar_opciones([
        ("1", "Evaluar modelo"),
        ("2", "Comparar modelos"),
        ("3", "Inferencia sobre imagen"),
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
    model = elegir_opcion(modelos, "Introduce el numero o nombre del modelo: ")

    mostrar_seccion("Datasets disponibles")
    listar(datasets)
    dataset = elegir_opcion(datasets, "Introduce el numero o nombre del dataset: ")

    folds = int(input("Introduce el numero de folds: ").strip())
    configuration = input("Nombre de configuracion para guardar resultados [modelo]: ").strip()
    testAndsave(configuration, model, dataset, folds)


def comparar_modelos():
    metric_labels = {"Detection": "Deteccion", "Classification": "Clasificacion"}
    for metric_type, label in metric_labels.items():
        mostrar_seccion(f"Top F1 - {label}")
        resultados = compare_consolidated_results(metric="F1-Score", metric_type=metric_type, limit=10)
        if not resultados:
            print("Sin resultados disponibles")
            continue
        for index, row in enumerate(resultados, start=1):
            print(f"{index}. {row['configuration']} -> {row['value']:.4f}")


def inferencia_imagen():
    modelos = get_trained_models()
    if not modelos:
        print("No hay modelos entrenados en 'models/trained'.")
        return

    mostrar_seccion("Modelos entrenados")
    listar(modelos)
    model = elegir_opcion(modelos, "Introduce el numero o nombre del modelo: ")

    weights = get_model_weight_files(model)
    weight_options = [_format_weight_option(model, weight) for weight in weights]

    mostrar_seccion("Pesos disponibles")
    listar(weight_options)
    selected_weight_label = elegir_opcion(weight_options, "Introduce el numero o ruta de pesos: ")
    selected_weight = weights[weight_options.index(selected_weight_label)]

    image_path = input("Ruta de la imagen a analizar: ").strip().strip('"')
    if not image_path:
        raise ValueError("Debes indicar una imagen")

    confidence_input = input("Confianza minima [0.25]: ").strip()
    confidence = float(confidence_input) if confidence_input else 0.25
    if confidence < 0 or confidence > 1:
        raise ValueError("La confianza debe estar entre 0 y 1")

    output_path = input("Ruta de salida [tests/inference/...]: ").strip().strip('"')
    output_path = output_path or None

    result = run_image_inference(
        tmodel_name=model,
        weights_path=selected_weight,
        image_path=image_path,
        output_path=output_path,
        confidence=confidence,
    )

    print(f"\nImagen generada: {result['output_path']}")
    print(f"Detecciones: {len(result['detections'])}")
    for index, detection in enumerate(result["detections"], start=1):
        print(f"{index}. {detection['class']} -> {detection['confidence']:.2f}")


def _format_weight_option(model_name, weight_path):
    model_root = next(parent for parent in weight_path.parents if parent.name == model_name)
    return str(weight_path.relative_to(model_root))
