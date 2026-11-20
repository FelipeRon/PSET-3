# Revisión frente a CD PSet 3

Estado al 20 de noviembre de 2026. Esta copia corresponde a la corrida de Snowflake. Los datos, notebooks y corridas se verifican con el snapshot identificado en docs/dataset.json.

| Requisito | Evidencia | Estado |
|---|---|---|
| Leer OBT Snowflake | src/datos.py y dataset.json | Ejecutado desde Snowflake, con Query ID, hash y 10.000 lecturas |
| EDA previa, tamaños, tipos, vacíos y descriptivos | 01_eda, CSV y figuras | Ejecutado en train |
| IQR, boxplots, Z score absoluto >=3 | atipicos.csv, boxplots.png | Ejecutado |
| Relación con target y desbalance | Spearman justificado y eda.json | Ejecutado |
| Tres hallazgos y consecuencias | eda.json y memo | Documentado con evidencia train |
| Dos features derivadas y disponibilidad en inferencia | PhysicalFeatures, notebook 02 y ablación en 03 | Ejecutado; diferencia térmica y potencia |
| Pipeline y ColumnTransformer sin fuga | src/pipeline.py y tests | Cuatro pruebas aprobadas |
| Split único y CV justificada | split.json, 80/20 estratificado, tres folds fijos | Ejecutado sin inspección exploratoria de test |
| MLflow Docker, backend y artifact store | docker-compose.yml y README | Servidor MLflow ejecutado en Docker con SQLite y artefactos locales |
| Dos baselines antes de candidatos | baselines.json, 2 runs fase baseline | Ejecutado |
| Cinco familias default | candidatos.json y notebook 03 | Ejecutado con semilla; logística, RF, HGB, MLP, GaussianNB |
| ChatGPT o Claude mediante prompting | docs/llm/, llm.json y llm_predictions.csv | Ejecutado en ChatGPT web, 60 lecturas de train y tres chats independientes. Registrado en MLflow |
| Tres mejores con 30 configuraciones cada una | tuning.json y corridas.csv | 90 corridas hijas, 30 por padre |
| Parámetros, métricas, datos, commit y modelos con firma | MLflow y corridas.csv | 103 corridas del protocolo principal y una corrida de prompting. La copia no conserva el historial Git |
| Medias, std, brecha, tiempo y diagnóstico | Notebooks, JSON y memo | Ejecutado; warnings de MLP conservados |
| Diferenciar mejora de dispersión | tabla tuning y notebook 04 | Ninguna mejora supera std entre folds |
| Selección antes de test, dos finalistas una vez | decision_before_holdout.json, lock y seleccion.json | Ejecutado |
| Costo de negocio y umbral | src/metricas.py; OOF train y capacidad 12 % | Ejecutado bajo supuestos, sin afirmar calibración |
| Registry y aliases | respaldo MLflow y versiones 1 y 2 | Champion y challenger registrados |
| Carga limpia del alias y reproducción | notebook 05 y prueba en otra carpeta | Reproducción exacta de métricas, atol 1e-12 |
| Exportación con mlflow.search_runs | docs/corridas.csv | Ejecutado |
| Captura de comparación en MLflow | docs/evidencias/mlflow_snowflake.jpg | Captura de MLflow con los finalistas de Snowflake |
| Cinco notebooks Restart y Run All | ejecución con nbclient, kernel nuevo por notebook | Cinco ejecutados con salidas guardadas |
| Memo hasta seis páginas más anexos | PSet3_memo_Ron_Andrade.pdf | Seis páginas principales y cinco anexos actualizados |

## Reproducción verificada

Se construyó la imagen sin caché, se restauró el ZIP en una copia limpia aislada y se reprodujo la métrica del champion. Se verificaron ambos aliases y pasaron cuatro pruebas. docs/evidencias/reproduccion_portable.json conserva el resultado.

Las corridas históricas registran git_commit=unavailable. docs/evidencias/codigo_sha256.csv registra el hash de los archivos de código que acompaña esta entrega.
