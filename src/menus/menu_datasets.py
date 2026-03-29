from services.dataset_service import get_datasets, generateFolderStructure
from services.processing import generate_multichannel_structure, execute
from utils.tools import limpiar_consola

def menu_datasets(error = False):
    limpiar_consola()
    while True:
        if error:
            print("Se ha producido un error, intentalo de nuevo")
        menu_datasets_options()
        opcion = input("Selecciona una opción: ").strip()
        if opcion == "1":
            # extraer_dataset()
            print(get_datasets())
            input("\nPulsa ENTER para volver al menú")
            limpiar_consola()
        elif opcion == "2":
            # generar_dataset()
            try :
                ruta = input('Introduzca la ruta del archivo .zip:')
                folds = input('A continuación introduzca el numero de folds que desea generar')
                generateFolderStructure( ruta, int(folds))
                limpiar_consola()
            except:
                menu_datasets(True)
        elif opcion == "3":
            # listar_datasets()
            flujo()
            print('listar_datasets')
        elif opcion == "0":
            break
        else:
            print("[ERROR] Opción no válida")
        error = False

def menu_datasets_options():
    print("\n--- MENÚ DATASETS ---")
    print("1. Ver datasets disponibles")
    print("2. Generar dataset a partir de archivo .zip")
    print("3. Generar dataset RGB")
    print("0. Volver")

def flujo():
    execute()    
    # print(get_datasets())
    # d = input("Selecciona un dataset")

    # print("1. ASM")
    # print("2. Entropy")

    # t = input("Selecciona filtro de Texturas")

    # print("Filtro morfologicos para la binarización")
    # print("1. Apertura")
    # print("2. Cierre")

    # f = input("Selecciona Filtro para la binarización")

    # print("kernels disponibles")
    # print("1. 2x2")
    # print("2. 3x3")
    # print("3. 4x4")
    # print("4. 5x5")

    # k = input("Selecciona el kernel")


    