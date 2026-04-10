import os
import shutil
import zipfile
from utils import paths
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path
from datetime import date

# Variable que contiene el directorio actual
CURRENT_DIRECTORY = os.getcwd()
DATASET_FOLDER = 'data'
TODAY = date.today()

def generateFolderStructure(zip_path:str, k_fold_value:int) -> str:
    """
    Descomprime el archivo zip y prepara la estructura de carpetas.

    Args:
        zip_path (str): Ruta al archivo zip.
        k_fold_value (int): Número de particiones.

    Returns:
        str: Ruta de la carpeta extraída.
    """

    print(f"Creando la estructura de carpetas para k = {k_fold_value}")

    # Descomprimimos el zip que contiene las imagenes que contiene el siguiente formato
    # Labels / Marked / Unmarked
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        
        zip_ref.extractall(f'{DATASET_FOLDER}/raw')

    # Directorio en el que se encuentran las iamgenes
    unmarked_dir = f'{CURRENT_DIRECTORY}/{DATASET_FOLDER}/raw/50 Images/Unmarked'
    labels_dir = f'{CURRENT_DIRECTORY}/{DATASET_FOLDER}/raw/50 Images/Labels'    

    # Almacenamos las imagenes
    imagenes = [
        f for f in os.listdir(unmarked_dir)
        if f.upper().endswith(('.JPG', '.PNG'))
        and os.path.exists(
            os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')
        )
    ]

    print(f"Estas son las imagenes {imagenes}")

    # Creamos el kfold
    kf = KFold(n_splits=k_fold_value, shuffle=True, random_state=20)

    n_split = 1

    dest_path = Path(Path(f"{CURRENT_DIRECTORY}/{DATASET_FOLDER}/processed") / f"{TODAY}_{k_fold_value}-Fold_Cross-val")
    dest_path.mkdir(parents=True, exist_ok=True)
    print(kf)

    for train_idx, val_idx in kf.split(imagenes):
        split_dir = f'{dest_path}/split_{n_split}'
        os.makedirs(split_dir, exist_ok=True)
        
        # Imágenes para test (10 imágenes en cada split)
        test_images = [imagenes[i] for i in val_idx]

        # Dividir train_val en train y val dentro de cada fold
        train_val_images = [imagenes[i] for i in train_idx]
        
        # De los train_val, seleccionamos 5 para validación y 35 para entrenamiento
        train_images, val_images = train_test_split(train_val_images, test_size=5, random_state=20)

        # Paso 4: Copiar las imágenes y etiquetas a las carpetas correspondientes
        def copy_files(image_list, subset):
            os.makedirs(os.path.join(split_dir, subset, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, subset, 'labels'), exist_ok=True)
            for img_name in image_list:
                # Copiar la imagen
                shutil.copy(os.path.join(unmarked_dir, img_name), os.path.join(split_dir, subset, 'images', img_name))
                # Copiar la etiqueta correspondiente
                label_name = img_name.replace('.JPG', '.txt').replace('.PNG', '.txt')
                shutil.copy(os.path.join(labels_dir, label_name), os.path.join(split_dir, subset, 'labels', label_name))

        # Copiar las imágenes de entrenamiento y sus etiquetas
        copy_files(train_images, 'train')

        # Copiar las imágenes de validación y sus etiquetas
        copy_files(val_images, 'val')

        # Copiar las imágenes de prueba y sus etiquetas al conjunto de test
        copy_files(test_images, 'test')

        # Paso 5: Crear el archivo .yaml
        yaml_content = f"""
            names:
            - NoDicentrico
            - Dicentrico
            path: {split_dir}
            train: train
            val: val
            test: test
        """

        # Guardar el archivo .yaml en el directorio del split
        with open(os.path.join(split_dir, 'data.yaml'), 'w') as yaml_file:
            yaml_file.write(yaml_content)

        print(f"Split {n_split} creado con éxito.")
        n_split += 1

def get_datasets():
    data_path = Path(f"{paths.DATA_DIR}/processed")

    if not data_path.exists():
        raise FileNotFoundError(f"No existe la carpeta {paths.DATA_DIR}")

    datasets = [d.name for d in data_path.iterdir() if d.is_dir()]

    return datasets


if __name__ == "__main__":
    generateFolderStructure("50Images.zip", 5)