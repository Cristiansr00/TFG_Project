from services.dataset_service import get_datasets
from services.models_service import get_YOLO_models
from services.training import training_model
from utils import paths
from utils.tools import limpiar_consola


def menu_modelos():
    while True:
        # limpiar_consola()
        menu_modelos_options()

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            # entrenar_modelo()
            print("entrenar_modelo")
            print(get_YOLO_models())
            yolo_model = input("Introduce el modelo YOLO a utilizar: ").strip()
            print("Tambien puedes utilizar otros modelos no descargados")
            flujo(yolo_model=yolo_model)
        elif opcion == "2":
            # cargar_modelo()
            print("cargar_modelo")
        elif opcion == "3":
            # listar_modelos()
            print("listar_modelos")
        elif opcion == "0":
            break
        else:
            print("[ERROR] Opción no válida")

def menu_modelos_options():
    print("\n--- MENÚ MODELOS ---")
    print("1. Entrenar modelo")
    print("2. Cargar modelo")
    print("3. Listar modelos")
    print("0. Volver")


def flujo(yolo_model:str="yolov8x"):
    # print("Selecciona el modelo a entrenar:")
    # print("1. Modelo A")
    # print("2. Modelo B")
    # m = input("Introduce el número del modelo:")

    print("Selecciona el dataset a utilizar:")
    print(get_datasets())
    dataset = input("Introduce el número del dataset:")
    print("Introduce el número de epochs:")
    e = input()
    print("Introduce el numero de folds aprendizaje:")
    lr = input()

    print(f"Entrenando modelo models/YOLO/{yolo_model}.pt con dataset {dataset}, epochs: {e}, learning rate: {lr}")
    
    # yolo_model_exists(f"{paths.MODELS_YOLO_DIR}/{yolo_model}.pt", yolo_model)
    training_model(e, lr, dataset, yolo_model=f"{paths.MODELS_YOLO_DIR}/{yolo_model}.pt")