# PSet 3 Experimentación de falla actual con AI4I

Autor: Felipe Ron Andrade. Continuación de [PSet 2](https://github.com/FelipeRon/Pset-02), commit `6ebc989edeaa0a60fa408361b6832ad92438c6ab`.

La fuente principal es `PSET2.OBT.OBT_READINGS`. El alcance es detectar **falla en la lectura actual**, sin atribuir a AI4I historia por máquina, predicción futura o RUL. Se comparan dos baselines, cinco familias y 90 configuraciones de tuning. El registro MLflow conserva champion y challenger.

## Evidencia incluida

Las salidas guardadas corresponden a **UCI oficial y reconstrucción local del contrato OBT**, no a una consulta autenticada de Snowflake. El autor confirmó que la OBT existe, pero su llave permanece en Codespaces. `docs/dataset.json` fija esta procedencia y su hash. No cambiar la etiqueta de procedencia sin ejecutar de nuevo desde Snowflake.

Los cinco notebooks tienen salidas reales. `docs/corridas.csv` se exporta con `mlflow.search_runs`. `docs/mlflow_snapshot.zip` contiene un respaldo portable de tracking SQLite, modelos completos, firmas y aliases. La prueba real de ChatGPT o Claude requiere API key privada y modelo explícito; sin ellos `docs/llm.json` indica `not_executed`, sin métricas inventadas. Consultar `docs/cumplimiento.md`.

## Preparación

Requisitos: Git, Docker Compose v2, 4 GB de RAM o más. Para ejecutar en host usar Python 3.12. Las 90 configuraciones con tres folds pueden tardar varios minutos.

```bash
git clone https://github.com/FelipeRon/PSET-3.git
cd PSET-3
cp .env.example .env
mkdir -p artifacts secrets
docker compose build
```

Completar `.env` solo localmente. No se necesitan las contraseñas de Kestra o Postgres del PSet 2. Mantener privado el puerto 5000 de Codespaces.

## Restaurar y reproducir la entrega

Restaurar **antes** de iniciar MLflow; el comando no sobrescribe un tracking existente:

```bash
docker compose run --rm --no-deps runner python -m src.portable restore
docker compose up -d mlflow
docker compose run --rm runner python -m src.experimentos verify
docker compose run --rm runner python scripts/execute_notebooks.py
```

Abrir `http://localhost:5000` o el puerto reenviado 5000 de Codespaces. Seleccionar `ai4i-mantenimiento`; revisar padres/hijas y Models → `ai4i-mantenimiento-modelo`, con aliases champion y challenger. El respaldo incluye el snapshot para reproducir exactamente la métrica. La restauración usa artefactos locales: ejecutar los clientes dentro del runner, con el mismo montaje `/app`; no desde otra ruta host. En experimentación nueva el servidor utiliza artifact serving HTTP.

## Nueva experimentación desde Snowflake

Usar un checkout nuevo, sin restaurar el respaldo, con `data/` y `artifacts/` vacíos. Conservar la entrega anterior en su carpeta.

1. Copiar la llave autorizada del PSet 2 a `secrets/snowflake_key.p8`. No crear otras credenciales si las existentes funcionan.
2. Completar `.env`: cuenta, usuario `PSET2_SVC`, rol `PSET2_ROLE`, warehouse `PSET2_WH`, base `PSET2` y `SNOWFLAKE_PRIVATE_KEY_FILE=/run/secrets/snowflake_key.p8`.
3. Mantener `DATA_SOURCE=snowflake` y `SNOWFLAKE_OBT_TABLE=PSET2.OBT.OBT_READINGS`.
4. Archivar los checkpoints de la entrega antes de crear el tracking nuevo; después ejecutar:

```bash
docker compose run --rm --no-deps runner python scripts/new_experiment.py
docker compose up -d mlflow
docker compose run --rm runner python scripts/execute_notebooks.py
```

Se extrae un snapshot una sola vez y se registra Query id, hash y fecha técnica. Cada notebook se ejecuta en kernel nuevo. Las fases terminadas se recuperan de checkpoints; no se vuelven a elegir ganadores sobre test. Un lock protege una evaluación final incompleta: recuperar esa corrida antes de volver a abrir el holdout.

También se puede ejecutar sin notebooks:

```bash
docker compose run --rm runner python -m src.experimentos all
docker compose run --rm runner python -m src.llm
docker compose run --rm runner python -m src.explain
docker compose run --rm runner python scripts/build_report.py
```

## Notebooks y ejecución con Python

Orden: `01_eda.ipynb`, `02_features.ipynb`, `03_baseline_candidatos.ipynb`, `04_tuning.ipynb`, `05_seleccion.ipynb`. Todos importan la preparación de `src/pipeline.py`. Para revisión manual usar Restart Kernel y Run All en ese orden.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python scripts/execute_notebooks.py
```

En Windows activar `.venv\Scripts\Activate.ps1`. En host, cambiar la ruta de la llave a su ruta local; `/run/secrets` es solo para Docker. No es necesario regenerar los notebooks con `build_notebooks.py`, que es una utilidad de desarrollo. Para ensayo sin cuenta, `DATA_SOURCE=uci_replay` descarga el ZIP oficial con TLS y conserva su procedencia explícita.

## Diseño experimental

- Split estratificado 80/20, semilla 42, persistido por UDI antes de EDA. EDA usa solo train; no hay tiempo o grupos reales para justificar otro esquema.
- Tres folds estratificados fijos. Imputación, one hot y escalado dentro de Pipeline y ColumnTransformer.
- Lista blanca de seis sensores/tipo. Se excluyen target, subtipos de falla, baseline_alert, identificadores, hash y quality_tier redundante.
- Features físicas: diferencia térmica y potencia. EDA aporta evidencia train; la ablación se ejecuta después de las dos líneas base.
- Técnica: Average Precision (AP), por desbalance. Negocio: costo por 1.000 lecturas con Cp=500, Ce=8.000, Cr=1.500 USD como supuestos, no costos observados.
- Baselines: DummyClassifier de clase mayoritaria y regla desgaste >=200 o torque >=60. Cinco familias: logística, Random Forest, HistGradientBoosting, MLP y GaussianNB. Defaults salvo semilla; se conservan warnings de convergencia.
- Las tres mejores familias por AP de CV reciben 30 configuraciones aleatorias cada una. Cada configuración registra pipeline, firma, ejemplo, parámetros, fuente, semilla, commit, métricas y tiempos.
- Champion y challenger son de familias diferentes, congelados por CV antes de test. El umbral minimiza costo sobre probabilidades OOF de train con máximo 12 % de alertas en ese conjunto. Se reporta la tasa real de test; no se garantiza capacidad futura ni calibración.
- Test se evalúa una vez por finalista. Las verificaciones posteriores reproducen el alias sin cambiar decisiones. Notebook 05 exige igualdad de métricas con tolerancia 1e-12.

## ChatGPT o Claude mediante prompting

La clasificación numérica admite prompting; no se declara imposible. Configurar en `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=nombre-del-modelo-autorizado
OPENAI_API_KEY=solo-en-tu-env-local
```

Para Claude: `LLM_PROVIDER=anthropic`, `LLM_MODEL` autorizado y `ANTHROPIC_API_KEY`. Ejecutar `python -m src.llm`. Usa 60 lecturas naturales de folds de train y ejemplos exclusivos del train de cada fold; nunca envía holdout. Registra prompt, respuestas, modelo, tiempo y métricas. No participa en el ranking principal porque su muestra y presupuesto son diferentes. Sin API configurada se registra el pendiente; no se fabrica evidencia. No pegar claves en notebooks ni GitHub.

## Entregables y límites

El memo `docs/PSet3_memo_Ron_Andrade.pdf` tiene seis páginas principales y anexos. Los JSON, CSV y figuras en docs contienen las mediciones; `dataset.json` y `split.json` fijan el snapshot. `mlflow_snapshot.zip` permite restaurar el registro sin acceso al servidor original.

Pruebas: `python -m pytest tests -q`. Actualizar gráficos y memo tras una corrida nueva: `python scripts/build_report.py`. La captura MLflow incluida corresponde a la corrida local; debe reemplazarse si se presenta una corrida cloud distinta.

AI4I es sintético y describe falla actual. No demuestra ahorro real ni transferencia a planta. Reutilizar folds para tuning puede dar optimismo; no equivale a CV anidada. El LLM sigue pendiente hasta configurar API. La infraestructura Docker queda preparada; su build en Codespaces requiere comprobación en ese entorno. Ninguna limitación se oculta mediante resultados simulados.
