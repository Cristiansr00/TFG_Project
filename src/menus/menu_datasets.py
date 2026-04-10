from services.dataset_service import get_datasets, generateFolderStructure
from services.processing import generate_multichannel_structure, execute, execute2
from utils.tools import limpiar_consola, color, ROJO, VERDE, KERNELS, MORPHS
from utils.TransformType import get_transformType, print_transform_types

def menu_datasets(error = False):
    limpiar_consola()
    while True:
        if error:
            print("Se ha producido un error, intentalo de nuevo")
        menu_datasets_options()
        opcion = input("Selecciona una opción: ").strip()
        if opcion == "1":
            # extraer_dataset()
            print(color(get_datasets(), ROJO))
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
    # execute2()    
    print(color(get_datasets(), ROJO))
    dataset = input("Selecciona un dataset")
    
    transforms = {}

    for c in ["r", "g", "b"]:
        print(f"Introduce la siguiente información para el canal {c}")
        transform = select_transform(c)
        transforms[c] = transform

    print("\n--- RESUMEN DE TRANSFORMACIONES ---")
    print(f"Dataset: {dataset}")
    for canal, transform in transforms.items():
        print(f"Canal {canal.upper()}: {transform}")
    
    confirmacion = input("\n¿Deseas continuar? (s/n): ").lower().strip()
    if confirmacion != "s":
        print("Operación cancelada")
        return
    
    # Generar estructura con las transformaciones seleccionadas
    try:
        dest_path = generate_multichannel_structure(
            dataset,
            r=transforms["r"],
            g=transforms["g"],
            b=transforms["b"]
        )
        print(color(f"✓ Dataset multichannel generado exitosamente {dest_path}", VERDE))
    except Exception as e:
        print(color(f"✗ Error al generar dataset: {e}", ROJO))


def print_kernels():
    for numero, dato in KERNELS.items():
        print(f"{numero}: {dato["name"]}")


def print_morphs():
    for numero, dato in MORPHS.items():
        print(f"{numero}: {dato["name"]}")
        

def select_transform(canal):
    print_transform_types()
    transform = input(f"Seleccinoa una de las transformaciones para el canal {canal}")

    transform_type = get_transformType(transform)
    print(f"{transform_type}")

    if transform_type:
        if transform_type == "Binary":
            # Preguntamos por el filtro morfologico
            print_morphs()
            filter = input(f"Inserta el tipo de filtro")
            # ask for a kernel and filter
            print_kernels()
            kernel = input(f"Inserta el tipo de kernel:")

            return {"type":int(transform), "kernel": int(kernel), "morph": int(filter)}
    return {"type":int(transform)}



