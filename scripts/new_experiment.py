"""Archiva checkpoints entregados antes de una corrida Snowflake nueva."""
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.datos import DOCS

if (ROOT/'data/snapshot.parquet').exists() or (ROOT/'artifacts/mlflow.db').exists():
    raise SystemExit('Se necesita un checkout sin snapshot ni tracking. Conservar la corrida anterior; no se borra.')
archive=ROOT/'artifacts/reference_docs'
if archive.exists():raise SystemExit('Los checkpoints ya fueron archivados. Continuar con la ejecución nueva.')
archive.mkdir(parents=True)
names=['dataset.json','split.json','eda.json','baselines.json','candidatos.json','ablacion.json',
       'plan_tuning.json','tuning.json','tuning_hist_boosting.json','tuning_random_forest.json','tuning_mlp.json',
       'tuning_logistic.json','tuning_gaussian_nb.json','seleccion.json','decision_before_holdout.json',
       'tracking_audit.json','llm.json','explicacion.json','corridas.csv','champion_predictions.csv',
       'challenger_predictions.csv','llm_predictions.csv','importancias_train.csv']
for name in names:
    path=DOCS/name
    if path.exists():shutil.move(str(path),str(archive/name))
print('Checkpoints de referencia archivados; la nueva ejecución generará su propio split y resultados.')
