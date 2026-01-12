import os
import zipfile
from sklearn.model_selection import KFold

# Variable que contiene el directorio actual
CURRENT_DIRECTORY = os.getcwd()
DATASET_FOLDER = 'dataset'


def generateFolderStructure(zip_path:str, k_fold_value:int) -> str:
  """
  Descomprime el archivo zip y prepara la estructura de carpetas.

  Args:
      zip_path (str): Ruta al archivo zip.
      k_fold_value (int): Número de particiones.

  Returns:
      str: Ruta de la carpeta extraída.
  """
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        
        zip_ref.extractall(DATASET_FOLDER)

  kf = KFold(n_splits=k_fold_value, shuffle=True, random_state=20)

  print(kf)

  for train_idx, val_idx in kf.spplit():

if __name__ == "__main__":
    generateFolderStructure("50Images.zip", 4)