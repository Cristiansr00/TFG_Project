from services.dataset_service import get_datasets
from services.models_service import get_YOLO_models, get_trained_models, resolve_yolo_model
from services.training import training_model
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


def menu_modelos():
    while True:
        menu_modelos_options()
        opcion = leer_opcion()

        if opcion == "1":
            modelos = get_trained_models()
            if not modelos:
                print("\nNo hay modelos entrenados en 'models/trained'.")
            else:
                mostrar_seccion("Modelos entrenados disponibles")
                listar(modelos)

            pausar("Pulsa ENTER para volver al menú...")
        elif opcion == "2":
            mostrar_seccion("Modelos YOLO instalados")
            listar(get_YOLO_models())
            yolo_model = input(
                "\nIntroduce el modelo YOLO a utilizar. Puedes escribir uno no instalado: "
            ).strip()
            # El entrenamiento puede tardar mucho; validamos entradas antes de lanzarlo.
            try:
                flujo(yolo_model=yolo_model)
            except ValueError as e:
                mostrar_error(str(e))
                pausar()
        elif opcion == "0":
            break
        else:
            mostrar_error("Opción no válida. Por favor, selecciona una opción válida.")
            pausar()


def menu_modelos_options():
    mostrar_titulo("MENÚ MODELOS")
    mostrar_opciones([
        ("1", "Listar modelos entrenados"),
        ("2", "Entrenar modelo"),
        ("0", "Volver"),
    ])


def flujo(yolo_model: str = "yolov8x"):
    mostrar_seccion("Selección de dataset")
    datasets = get_datasets()
    if not datasets:
        print("\nNo hay datasets en 'data/processed'.")
        return

    listar(datasets)
    dataset = elegir_opcion(datasets, "Introduce el número o nombre del dataset: ")

    epochs = input("Introduce el número de epochs: ").strip()
    folds = input("Introduce el número de folds de aprendizaje: ").strip()
    # Se resuelve a ruta local si el .pt existe; si no, Ultralytics intentará resolverlo.
    resolved_model = resolve_yolo_model(yolo_model)

    mostrar_seccion("Resumen del entrenamiento")
    print(f"Modelo: {resolved_model}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Folds: {folds}")
    training_model(epochs, folds, dataset, yolo_model=resolved_model)
