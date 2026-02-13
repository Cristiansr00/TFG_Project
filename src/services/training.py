import torch
from ultralytics import YOLO
import os
import utils.paths as paths

# CURRENT_DIRECTORY = os.getcwd()
# YOLO_MODEL_VERSION = 'yolo26m.pt'



def training_model(epochs, folds_number, dataset:str, yolo_model):
    epochs = int(epochs)
    folds_number = int(folds_number)
    # dir_base = f"{CURRENT_DIRECTORY}/{dataset}/2026-01-18_5-Fold_Cross-val"
    dir_base = f"{paths.DATA_DIR}/processed/{dataset}"
    for i in range(1, folds_number+1):
        split_path = f"split_{i}/data.yaml"

        project = f"{paths.MODELS_DIR}/MODEL_{dataset}/kfold_training(basic split_{i} {epochs}epochs)"
        training_session(project, f"{dir_base}/{split_path}", epochs, yolo_model)

def training_session(project, yaml_path, epochs, yolo_model):
    """
    Esta función inicia el proceso de entrenamiento de un modelo YOLOv8.
    Se encarga de crear la estructura de directorios necesaria y de gestionar los
    diferentes tipos de entrenamiento según los parámetros proporcionados.
    Los parámetros son:
    - project: Nombre de la carpeta donde se guardarán los resultados del entrenamiento.
    - yaml_path: Ruta al archivo YAML de configuración del dataset.
    - epochs: Número de épocas para el entrenamiento.
    """
    if os.path.exists(project):
        print(f"La ruta {project} ya existe. No se realizará el entrenamiento.")
        return
    
    model = YOLO(model=yolo_model, task="detect")
    print(f"Entrenando modelo YOLO con dataset en {yaml_path} durante {epochs} epochs.")
    model.to('cuda')
    results1 = model.train(
        data = yaml_path,
        epochs = epochs,
        imgsz=640,
        batch = 4,
        project = project,
        name = "train",
        device=0,
    )              
     # batch = 8 es necesario porque la GPU del server no soporta superior
    
    # Crear la ruta completa para el archivo de texto
    metrics_file_path = os.path.join(project, "training_metrics.txt")
        
    # Escribir las métricas en el archivo de texto
    with open(metrics_file_path, 'w') as f:
        f.write("Metricas para 'Dicentrico':\n")
        f.write(f" TP = {int(results1.confusion_matrix.matrix[1][1])}\n")  # Verdadero positivo (clase 1)
        f.write(f" FP = {int(results1.confusion_matrix.matrix[0][1])}\n")  # Falso positivo (clase 0 predicho como 1)
        f.write(f" FN = {int(results1.confusion_matrix.matrix[1][0])}\n")  # Falso negativo (clase 1 predicho como 0)
        
        f.write("Metricas para 'NoDicentrico':\n")
        f.write(f" TP = {int(results1.confusion_matrix.matrix[0][0])}\n")  # Verdadero positivo (clase 0)
        f.write(f" FP = {int(results1.confusion_matrix.matrix[1][0])}\n")  # Falso positivo (clase 1 predicho como 0)
        f.write(f" FN = {int(results1.confusion_matrix.matrix[0][1])}\n")  # Falso negativo (clase 0 predicho como 1)

        f.write("Otras metricas:\n")
        f.write(" Precision media para todas las clases: {}\n".format(results1.box.all_ap))
        f.write(" Precision media: {}\n".format(results1.box.ap))
        f.write(" Precision media a IoU=0.50: {}\n".format(results1.box.ap50))
        f.write(" F1 score: {}\n".format(results1.box.f1))
        f.write(" Media de precision media: {}\n".format(results1.box.map))
        f.write(" Media de precision media a IoU=0.50: {}\n".format(results1.box.map50))
        f.write(" Media de precision media a IoU=0.75: {}\n".format(results1.box.map75))
        f.write(" Media de precision media para diferentes umbrales de IoU: {}\n".format(results1.box.maps))
        f.write(" Media de precision: {}\n".format(results1.box.mp))
        f.write(" Media de recall: {}\n".format(results1.box.mr))
        f.write(" Precision: {}\n".format(results1.box.p))
        f.write(" Valores de precision: {}\n".format(results1.box.prec_values))
        f.write(" Recall: {}\n".format(results1.box.r))

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    training_model(100, 5, 'multichannel') 