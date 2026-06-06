from services.dataset_service import get_datasets, generateFolderStructure
from services.processing import generate_monochannel_structure, generate_multichannel_structure
from utils.tools import (
    KERNELS,
    MORPHS,
    elegir_opcion,
    leer_opcion,
    listar,
    mostrar_error,
    mostrar_exito,
    mostrar_opciones,
    mostrar_seccion,
    mostrar_titulo,
    pausar,
)
from utils.TransformType import TransformType, get_transformType, print_transform_types


def menu_datasets(error=False):
    while True:
        menu_datasets_options()
        if error:
            mostrar_error("Se ha producido un error, inténtalo de nuevo.")
        opcion = leer_opcion()

        if opcion == "1":
            datasets = get_datasets()
            if datasets:
                mostrar_seccion("Datasets disponibles")
                listar(datasets)
            else:
                print("No hay datasets en 'data/processed'.")
            pausar("Pulsa ENTER para volver al menú...")
        elif opcion == "2":
            try:
                mostrar_seccion("Generar estructura k-fold")
                ruta = input("Introduce la ruta del archivo .zip: ").strip()
                folds = int(input("Introduce el número de folds que deseas generar: ").strip())
                generateFolderStructure(ruta, folds)
            except Exception as e:
                mostrar_error(f"Error al generar la estructura: {e}")
                pausar()
        elif opcion == "3":
            try:
                flujo()
            except (ValueError, FileNotFoundError) as e:
                mostrar_error(f"Error al generar dataset: {e}")
                pausar()
        elif opcion == "0":
            break
        else:
            mostrar_error("Opción no válida.")
            pausar()
        error = False


def menu_datasets_options():
    mostrar_titulo("MENÚ DATASETS")
    mostrar_opciones([
        ("1", "Ver datasets disponibles"),
        ("2", "Generar dataset a partir de archivo .zip"),
        ("3", "Generar dataset sintético"),
        ("0", "Volver"),
    ])


def flujo():
    datasets = get_datasets()
    if not datasets:
        print("No hay datasets en 'data/processed'.")
        return

    mostrar_seccion("Dataset base")
    listar(datasets)
    dataset = elegir_opcion(datasets, "Selecciona el número o nombre del dataset base: ")

    mostrar_seccion("Tipo de dataset sintético")
    mostrar_opciones([
        ("1", "Monocanal"),
        ("2", "Multicanal (RGB)"),
    ])
    tipo = leer_opcion()

    if tipo == "1":
        mostrar_seccion("Canal único")
        transform = select_transform("mono")

        mostrar_seccion("Resumen de transformaciones")
        print(f"Dataset: {dataset}")
        print(f"Canal único: {transform}")
        confirmar = input("\n¿Deseas continuar? (s/n): ").lower().strip()
        if confirmar != "s":
            print("Operación cancelada")
            return

        # El servicio detecta los folds desde split_N y genera el nombre del dataset final.
        dest_path = generate_monochannel_structure(dataset, transform)
        mostrar_exito(f"Dataset monocanal generado exitosamente: {dest_path}")
        pausar()

    elif tipo == "2":
        transforms = {}
        for canal in ["r", "g", "b"]:
            mostrar_seccion(f"Canal {canal.upper()}")
            transforms[canal] = select_transform(canal)

        mostrar_seccion("Resumen de transformaciones")
        print(f"Dataset: {dataset}")
        for canal, transform in transforms.items():
            print(f"Canal {canal.upper()}: {transform}")

        confirmar = input("\n¿Deseas continuar? (s/n): ").lower().strip()
        if confirmar != "s":
            print("Operación cancelada")
            return

        # Cada canal RGB puede tener una transformación distinta.
        dest_path = generate_multichannel_structure(
            dataset,
            r=transforms["r"],
            g=transforms["g"],
            b=transforms["b"],
        )
        mostrar_exito(f"Dataset multicanal generado exitosamente: {dest_path}")
        pausar()
    else:
        raise ValueError("Tipo de dataset no válido")


def print_kernels():
    mostrar_seccion("Kernels disponibles")
    for numero, dato in KERNELS.items():
        print(f"  {numero}. {dato['name']}")


def print_morphs():
    mostrar_seccion("Filtros morfológicos disponibles")
    for numero, dato in MORPHS.items():
        print(f"  {numero}. {dato['name']}")


def _read_int_option(prompt, opciones):
    value = input(prompt).strip()
    if not value.isdigit():
        raise ValueError(f"Valor no numérico: {value}")
    value = int(value)
    if value not in opciones:
        raise ValueError(f"Opción no válida: {value}")
    return value


def select_transform(canal):
    print_transform_types()
    transform = _read_int_option(
        f"Selecciona una transformación para el canal {canal}: ",
        {item.value for item in TransformType},
    )

    transform_type = get_transformType(transform)
    if transform_type is None:
        raise ValueError(f"Transformación no válida: {transform}")

    config = {"type": TransformType(transform)}
    if transform_type == "Binary":
        print_morphs()
        morph = _read_int_option("Inserta el tipo de filtro morfológico: ", MORPHS.keys())
        print_kernels()
        kernel = _read_int_option("Inserta el tipo de kernel: ", KERNELS.keys())
        config.update({"kernel": kernel, "morph": morph})

    return config
