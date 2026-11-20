# PSet 3 · Experimentación y selección del modelo

El modelo detecta si hay **una falla en la lectura actual** de una máquina a partir de los sensores y el tipo de producto del conjunto AI4I. La fuente es `PSET2.OBT.OBT_READINGS`, la tabla que construye el pipeline del [PSet 2](https://github.com/FelipeRon/Pset-02) en Snowflake.

```text
OBT (Snowflake) → Snapshot → EDA → Features → Líneas base → Candidatos → Tuning → Selección → MLflow Model Registry
```

## Estructura

| Ruta | Contenido |
| --- | --- |
| `Dockerfile`, `docker-compose.yml` | Entorno Python 3.12 y servidor MLflow con backend SQLite y almacenamiento local de artefactos. |
| `src/datos.py` | Consulta la OBT, valida sus columnas y crea el snapshot y la división estratificada. |
| `src/pipeline.py`, `src/metricas.py` | Features físicas, preprocesamiento, métricas y selección del umbral por costo. |
| `src/experimentos.py` | Entrena, registra y verifica modelos en MLflow. |
| `scripts/new_experiment.py`, `scripts/execute_notebooks.py` | Archivan los checkpoints de referencia y ejecutan los notebooks en orden. |
| `notebooks/01…05` | EDA, variables, líneas base y candidatos, tuning y selección. |

## 1. Preparar el entorno

Necesitas Docker con Docker Compose v2. Para entrenar con datos nuevos también necesitas acceso de lectura a `PSET2.OBT.OBT_READINGS`. Ejecuta los comandos desde la raíz de PSet 3. Docker instala Python y usa las versiones registradas en `requirements.lock` al resolver las dependencias.

Para **entrenar desde Snowflake**, usa una copia nueva del proyecto con `data/` y `artifacts/` vacíos. Asigna a la copia un nombre de proyecto Compose y un puerto local distintos si ya hay otra instancia de MLflow activa. Para **verificar un modelo ya registrado**, usa la carpeta de su corrida y ve a la sección 5.

## 2. Configurar Snowflake

En una copia nueva, crea `.env` a partir de la plantilla:

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
```

Completa `SNOWFLAKE_ACCOUNT` y verifica las demás variables con la configuración de tu PSet 2:

```dotenv
DATA_SOURCE=snowflake
SNOWFLAKE_USER=PSET2_SVC
SNOWFLAKE_ROLE=PSET2_ROLE
SNOWFLAKE_WAREHOUSE=PSET2_WH
SNOWFLAKE_DATABASE=PSET2
SNOWFLAKE_OBT_TABLE=PSET2.OBT.OBT_READINGS
SNOWFLAKE_PRIVATE_KEY_FILE=/run/secrets/snowflake_key.p8
```

Reutiliza la clave privada del usuario de servicio. En `docker-compose.yml`, el montaje de `runner` que termina en `:/run/secrets:ro` debe apuntar a la carpeta del host que contiene `snowflake_key.p8`. Por defecto Compose monta `./secrets`. Coloca allí `snowflake_key.p8` o define `SNOWFLAKE_SECRETS_DIR` en `.env` con la ruta local de la carpeta que ya contiene la clave. Mantén `.env` y la clave privada fuera del control de versiones.

## 3. Construir el entorno e iniciar MLflow

En una copia nueva, ejecuta en este orden:

```powershell
docker compose config --quiet
docker compose build runner
docker compose run --rm --no-deps runner python scripts/new_experiment.py
docker compose up -d --no-build --wait mlflow
```

Ejecuta `new_experiment.py` una sola vez, antes de arrancar MLflow por primera vez. El script archiva los checkpoints incluidos en la copia para que el pipeline genere los suyos a partir de Snowflake.

En esta copia, abre [MLflow en 127.0.0.1:5003](http://127.0.0.1:5003). SQLite y los artefactos se guardan en `artifacts/mlflow.db` y `artifacts/mlruns/`. Compose configura el acceso a MLflow para el runner.

## 4. Ejecutar los notebooks

```powershell
docker compose run --rm runner python scripts/execute_notebooks.py
```

El script ejecuta cada notebook en un kernel nuevo y guarda sus salidas. Para ejecutarlos manualmente en un entorno con acceso al tracking, usa **Restart Kernel and Run All** en este orden.

| Orden | Notebook | Qué hace |
| --- | --- | --- |
| 1 | `01_eda.ipynb` | Consulta Snowflake, guarda el snapshot y explora el conjunto de entrenamiento. |
| 2 | `02_features.ipynb` | Construye las variables derivadas y verifica qué columnas puede recibir el modelo. |
| 3 | `03_baseline_candidatos.ipynb` | Ejecuta las líneas base, la ablación y la comparación de familias de modelos. |
| 4 | `04_tuning.ipynb` | Ajusta tres familias mediante muestreo aleatorio y registra 30 configuraciones por familia como corridas hijas en MLflow. |
| 5 | `05_seleccion.ipynb` | Selecciona los finalistas por validación cruzada, ajusta los umbrales con predicciones fuera de fold y evalúa los finalistas en el conjunto de prueba. Registra los aliases `champion` y `challenger`, exporta `docs/corridas.csv` y verifica el modelo registrado. |

El primer notebook guarda `data/snapshot.parquet`. Las fases siguientes reutilizan ese archivo. `docs/dataset.json` registra el identificador de la consulta, la fecha y el hash SHA-256 del dataset. `docs/split.json` conserva la división 80/20 por `reading_id`. La división y la validación cruzada estratificada de tres folds usan la semilla 42. La imputación, la codificación y el escalado se ajustan dentro del pipeline de cada fold.

## 5. Reproducir el modelo registrado

En otra computadora, abre una **copia limpia** del proyecto. Conserva `docs/mlflow_snapshot.zip`, `docs/dataset.json`, `docs/split.json` y `docs/seleccion.json`. El ZIP contiene la base de tracking, los artefactos y el snapshot. Puedes usar `.env.example` tal cual para esta comprobación offline respecto de Snowflake.

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
docker compose build runner
docker compose run --rm --no-deps runner python -m src.portable restore
docker compose up -d --no-build --wait mlflow
docker compose run --rm runner python -m src.experimentos verify
docker compose run --rm runner python -m pytest -q
```

En macOS o Linux, usa `cp .env.example .env` en lugar de `Copy-Item`. Ejecuta `restore` antes de iniciar MLflow por primera vez. Si ya restauraste el respaldo, continúa desde `docker compose up`. Si ejecutas dos copias en el mismo equipo, asigna un `COMPOSE_PROJECT_NAME` distinto a cada una y fija otro puerto con `MLFLOW_PORT` en `.env`.

El comando carga `models:/ai4i-mantenimiento-modelo@champion`, valida el hash del snapshot y recalcula las métricas sobre la misma partición de prueba. Si coinciden con las registradas en `docs/seleccion.json`, devuelve `reproduced: True`.

En la interfaz de MLflow, abre el experimento `ai4i-mantenimiento` para inspeccionar las corridas y el modelo `ai4i-mantenimiento-modelo` para consultar los aliases.

## Prueba exploratoria con ChatGPT

Los prompts enviados, las respuestas JSON y las etiquetas de corrección están en `docs/llm/`. Se usaron tres chats independientes, cada uno con seis ejemplos del entrenamiento de su fold y veinte consultas del segmento de validación del mismo fold. Las etiquetas de consulta se incorporan solo al calcular las métricas. El conjunto de prueba queda fuera del protocolo.

El script `scripts/prepare_chatgpt.py` prepara la muestra. `scripts/evaluate_chatgpt.py` valida las respuestas conservadas, calcula métricas, crea una corrida en MLflow y actualiza `docs/corridas.csv`. La interfaz web utilizada no mostró la versión interna del modelo.
