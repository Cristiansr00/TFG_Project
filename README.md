# TFG Project

Proyecto para generar datasets sinteticos a partir de transformaciones de imagen,
entrenar modelos YOLO con validacion cruzada y consolidar metricas de deteccion y
clasificacion.

## Estructura

- `src/main.py`: entrada de la aplicacion por consola.
- `src/menus`: menus interactivos para modelos, datasets y metricas.
- `src/services`: logica de generacion de datasets, entrenamiento y evaluacion.
- `src/transforms`: transformaciones originales, texturas y binarizaciones.
- `src/utils`: rutas, utilidades de consola y catalogo de transformaciones.
- `data/processed`: datasets generados o preparados para entrenamiento.
- `models/YOLO`: pesos base YOLO (`.pt`).
- `models/trained`: entrenamientos por dataset y fold.
- `tests`: resultados de evaluacion y Excel consolidado.

## Instalacion

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si vas a crear un entorno nuevo:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecucion

```powershell
.\venv\Scripts\python.exe src\main.py
```

El menu principal permite:

1. Gestionar modelos entrenados y lanzar nuevos entrenamientos.
2. Listar datasets, crear estructura k-fold desde un `.zip` y generar datasets sinteticos.
3. Evaluar modelos y comparar resultados consolidados por F1-score.

## Flujo recomendado

1. Coloca pesos base YOLO en `models/YOLO` o escribe un nombre compatible con Ultralytics.
2. Prepara o selecciona un dataset de `data/processed`.
3. Genera versiones monocanal o RGB desde el menu de datasets.
4. Entrena desde el menu de modelos indicando dataset, epochs y folds.
5. Evalua desde el menu de metricas.
6. Consulta la comparacion de modelos para ver los mejores F1 de deteccion y clasificacion.

## Notas

- Las rutas internas se resuelven desde la raiz del proyecto, no desde el directorio actual.
- La evaluacion guarda resultados en CSV por modelo y en `tests/consolidated_results.xlsx`.
- El entrenamiento usa CUDA si esta disponible; en caso contrario usa CPU.
