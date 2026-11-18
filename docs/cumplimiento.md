# Revisión frente a CD PSet 3

| Requisito | Evidencia | Estado |
|---|---|---|
| Leer OBT Snowflake | src/datos.py, Query id y snapshot previstos | Implementado; repetir con llave privada del autor en Codespaces |
| EDA previa, tamaños, tipos, vacíos y descriptivos | 01_eda, CSV y figuras | Ejecutado en train |
| IQR, boxplots, Z score absoluto >=3 | atipicos.csv, boxplots.png | Ejecutado |
| Relación con target y desbalance | Spearman justificado y eda.json | Ejecutado |
| Tres hallazgos y consecuencias | eda.json y memo | Documentado con evidencia train |
| Dos features derivadas y disponibilidad en inferencia | PhysicalFeatures, notebook 02 y ablación en 03 | Ejecutado; diferencia térmica y potencia |
| Pipeline y ColumnTransformer sin fuga | src/pipeline.py y tests | Cuatro pruebas aprobadas |
| Split único y CV justificada | split.json, 80/20 estratificado, tres folds fijos | Ejecutado sin inspección exploratoria de test |
| MLflow Docker, backend y artifact store | docker-compose.yml y README | Configuración preparada; servidor local real ejecutado fuera de Docker |
| Dos baselines antes de candidatos | baselines.json, 2 runs fase baseline | Ejecutado |
| Cinco familias default | candidatos.json y notebook 03 | Ejecutado con semilla; logística, RF, HGB, MLP, GaussianNB |
| ChatGPT o Claude mediante prompting | src/llm.py y llm.json | PENDIENTE llamada real; falta API key y modelo autorizado |
| Tres mejores con 30 configuraciones cada una | tuning.json y tracking_audit.json | 90 corridas hijas, 30 por padre |
| Parámetros, métricas, datos, commit y modelos con firma | MLflow y corridas.csv | 103 runs terminados, 100 modelos ajustados; 3 padres de agrupación |
| Medias, std, brecha, tiempo y diagnóstico | Notebooks, JSON y memo | Ejecutado; warnings de MLP conservados |
| Diferenciar mejora de dispersión | tabla tuning y notebook 04 | Ninguna mejora supera std entre folds |
| Selección antes de test, dos finalistas una vez | decision_before_holdout.json, lock y seleccion.json | Ejecutado |
| Costo de negocio y umbral | src/metricas.py; OOF train y capacidad 12 % | Ejecutado bajo supuestos, sin afirmar calibración |
| Registry y aliases | respaldo MLflow y versiones 1 y 2 | Champion y challenger registrados |
| Carga limpia del alias y reproducción | notebook 05 y prueba en otra carpeta | Reproducción exacta de métricas, atol 1e-12 |
| Exportación con mlflow.search_runs | docs/corridas.csv | Ejecutado |
| Captura de comparación en MLflow | docs/evidencias/mlflow_comparacion.png | PARCIAL; falta captura amplia con columnas y métricas legibles |
| Cinco notebooks Restart y Run All | ejecución con nbclient, kernel nuevo por notebook | Cinco ejecutados con salidas guardadas |
| Memo hasta seis páginas más anexos | PSet3_memo_Ron_Andrade.pdf | Seis páginas principales y dos anexos |

## Pendientes para cerrar la guía

1. Ejecutar el protocolo con DATA_SOURCE=snowflake en un checkout limpio, sin mezclar respaldo local, split o dataset de otro origen.
2. Configurar API privada y LLM_MODEL para la llamada real de prompting. El problema sí admite prompting: no se justifica omitirlo afirmando que es imposible.
3. Capturar la comparación completa desde la UI de MLflow y reemplazar la captura parcial. La automatización del navegador se detuvo al intentar ampliar la vista; no se fabricó otra captura.
4. Comprobar el build y ejecución de Compose en el Codespace del autor. La ejecución MLflow realizada aquí usa SQLite y artifact store locales con el mismo software fijado.

No se entregan métricas atribuidas a Snowflake, llamadas LLM, capturas completas o un build Docker que no se hayan ejecutado en esta sesión. El README permite completar esos pasos conservando la evidencia y sus limitaciones.
