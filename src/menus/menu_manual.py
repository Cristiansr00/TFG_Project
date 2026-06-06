from utils.tools import mostrar_seccion, mostrar_titulo, pausar


def menu_manual():
    mostrar_titulo("MANUAL DE USO")

    mostrar_seccion("Objetivo")
    print("Esta aplicación permite preparar datasets, entrenar modelos YOLO")
    print("y guardar métricas de evaluación desde consola.")

    mostrar_seccion("1. Gestión de modelos")
    print("  - Lista los modelos ya entrenados en models/trained.")
    print("  - Entrena un modelo YOLO sobre un dataset de data/processed.")
    print("  - Permite usar un .pt local de models/YOLO o un nombre compatible")
    print("    con Ultralytics.")

    mostrar_seccion("2. Gestión de datasets")
    print("  - Lista los datasets disponibles.")
    print("  - Genera una estructura k-fold desde un archivo .zip.")
    print("  - Crea datasets sintéticos monocanal o multicanal RGB.")
    print("  - Detecta automáticamente los folds desde las carpetas split_N.")

    mostrar_seccion("3. Métricas y evaluación")
    print("  - Evalúa un modelo entrenado sobre el conjunto test de cada fold.")
    print("  - Guarda resultados por modelo en CSV.")
    print("  - Actualiza tests/consolidated_results.xlsx.")
    print("  - Compara modelos ordenando por F1-score.")

    mostrar_seccion("Transformaciones")
    print("  - Original: mantiene la imagen en escala de grises.")
    print("  - Texturas: mean, std, contrast, dissimilarity, homogeneity,")
    print("    ASM, max, entropy y energy.")
    print("  - Binarización: Otsu y Spline con filtro morfológico y kernel.")

    mostrar_seccion("Flujo recomendado")
    print("  1. Prepara o selecciona un dataset base.")
    print("  2. Genera las configuraciones sintéticas que quieras evaluar.")
    print("  3. Entrena modelos sobre esos datasets.")
    print("  4. Evalúa los modelos.")
    print("  5. Consulta la comparación de métricas.")

    pausar("Pulsa ENTER para volver al menú principal...")
