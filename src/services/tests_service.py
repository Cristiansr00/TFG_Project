import csv
import os
from pathlib import Path
from ultralytics import YOLO
import utils.paths as paths
RUTA_METRICAS = 'tests'

def obtener_split_por_indice(splits_dir, i):
    splits_path = Path(splits_dir)
    patron = f"split_{i}"

    for d in splits_path.iterdir():
        if d.is_dir() and patron in d.name:
            return d


def testAndsave(configuration, tmodel_name, dataset_name, nFolds):
    """
    Esta función evalúa un modelo YOLOv8 entrenado en un conjunto de datos específico y guarda las métricas de rendimiento en un archivo CSV.
    Args:
        configuration (str): Configuración del modelo, puede ser una cadena vacía para la configuración `basic`.
        trainPath (str): Ruta al directorio del modelo YOLO ya entrenado.
        sD (str): Nombre del conjunto de datos utilizado para el entrenamiento.
        nFolds (int): Número de pliegues para la validación cruzada k-fold.
    Returns:
        None
    """

    # Fichero CSV para almacenar los resultados
    csv_file = f"{paths.TESTS_DIR}/{tmodel_name}/{dataset_name}/metrics_result.csv"
    csv_file_path = Path(paths.TESTS_DIR) / tmodel_name / dataset_name / "metrics_result.csv"
    
    # Comprueba si el archivo CSV ya existe y si la configuración ya está presente (no es necesario realmente)
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if configuration == "" and row[0] == "tmodel_name":
                    print(f"Configuration 'tmodel_name' already exists. Skipping.")
                    return
                elif row[0] == configuration:
                    print(f"Configuration '{configuration}' already exists. Skipping.")
                    return
    else:
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    TP_d = 0
    FP_d = 0
    FN_d = 0
        
    TP_c = 0
    TN_c = 0
    FP_c = 0
    FN_c = 0
    
    for j in range(1, nFolds + 1):
        # Actualiza la ruta del archivo data.yaml según el número de pliegue
        
        data_yaml = f"{paths.DATA_DIR}/processed/{dataset_name}/split_{j}/data.yaml"
        
        modifiedPath = obtener_split_por_indice(f"{paths.MODELS_DIR}/{tmodel_name}",j)
        # modifiedPath = tmodel_name.replace("split_", f"split_{j}")
        
        # Carga el modelo YOLOv8 entrenado para el pliegue actual
        model = YOLO(
            model=f"{paths.MODELS_DIR}/{tmodel_name}/{modifiedPath.name}/train/weights/best.pt"
        )
         
        # Ejecutar la validación en el conjunto de prueba y recuperar la matriz de confusión
        results = model.val(
            data=data_yaml,
            device="cpu",
            split="test"
        )
        confMatrix = results.confusion_matrix.matrix
        print(f"Confusion Matrix for fold {j}:\n{confMatrix}")
        # Extraer las métricas de la matriz de confusión
        TP_d += confMatrix[0][0] + confMatrix[0][1] + confMatrix[1][0] + confMatrix[1][1]
        FP_d += confMatrix[0][2] + confMatrix[1][2]
        FN_d += confMatrix[2][0] + confMatrix[2][1]
        
        TP_c += confMatrix[1][1]
        TN_c += confMatrix[0][0]
        FP_c += confMatrix[1][0]
        FN_c += confMatrix[0][1]
    
    # Calcular promedios
    TP_d /= nFolds
    FP_d /= nFolds
    FN_d /= nFolds
    TP_c /= nFolds
    TN_c /= nFolds
    FP_c /= nFolds
    FN_c /= nFolds
    
    # Calcular las métricas para la detección
    recall_d = TP_d / (TP_d + FN_d) if (TP_d + FN_d) > 0 else 0
    precision_d = TP_d / (TP_d + FP_d) if (TP_d + FP_d) > 0 else 0
    fmeasure_d = 2 * (precision_d * recall_d) / (precision_d + recall_d) if (precision_d + recall_d) > 0 else 0
    spatialacc_d = TP_d / (TP_d + FN_d + FP_d) if (TP_d + FN_d + FP_d) > 0 else 0
        
    # Calcular las métricas para la clasificación
    recall_c = TP_c / (TP_c + FN_c) if (TP_c + FN_c) > 0 else 0
    precision_c = TP_c / (TP_c + FP_c) if (TP_c + FP_c) > 0 else 0
    fmeasure_c = 2 * (precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0
    spatialacc_c = TP_c / (TP_c + FN_c + FP_c) if (TP_c + FN_c + FP_c) > 0 else 0
    accuracy_c = (TP_c + TN_c) / (TP_c + TN_c + FP_c + FN_c) if (TP_c + TN_c + FP_c + FN_c) > 0 else 0
    
    # Abrir el fichero en modo append para añadir nuevos resultados
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Escribe la cabecera si el fichero está vacío
        if file.tell() == 0:
            writer.writerow(["Configuration", "Metric Type", "TP", "TN", "FP", "FN", "Recall", "Precision", "F1-Score", "SpatialAccuracy", "Accuracy"])
        
        # Escribe los resultados de las métricas en el fichero CSV
        writer.writerow([configuration if configuration else f"{tmodel_name}", "Detection", float(TP_d), "-", float(FP_d), float(FN_d), recall_d, precision_d, fmeasure_d, spatialacc_d, "-"])
        writer.writerow([configuration if configuration else f"{tmodel_name}", "Classification", float(TP_c), float(TN_c), float(FP_c), float(FN_c), recall_c, precision_c, fmeasure_c, spatialacc_c, accuracy_c])