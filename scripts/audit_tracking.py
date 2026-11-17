"""Audita presupuesto y completa metadatos descriptivos de padres históricos."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import mlflow
from mlflow.tracking import MlflowClient
from src.experimentos import tracking, export_runs
from src.datos import DOCS, split, SEED, dump

tracking();rows=export_runs();_,_,meta=split();client=MlflowClient()
for _,row in rows[rows['tags.phase']=='tuning_parent'].iterrows():
    params={'seed':SEED,'cv_folds':3,'configurations':30,'source_table':meta['table'],
            'snapshot_utc':meta['snapshot_utc'],'dataset_sha256':meta['dataset_sha256']}
    existing=client.get_run(row.run_id).data.params
    for k,v in params.items():
        if k not in existing:client.log_param(row.run_id,k,v)
    client.set_tag(row.run_id,'metadata_note','Parent groups 30 child configurations; no separately fitted parent model')
rows=export_runs();counts=rows.groupby('tags.phase').size().to_dict()
assert counts['baseline']==2 and counts['candidato']==5 and counts['tuning']==90 and counts['final']==2
assert (rows.status=='FINISHED').all()
parents=rows[rows['tags.phase']=='tuning']
assert parents['tags.mlflow.parentRunId'].notna().all()
assert parents.groupby('tags.mlflow.parentRunId').size().eq(30).all()
fitted=rows[rows['tags.phase']!='tuning_parent']
for _,row in fitted.iterrows():
    model_path=Path(mlflow.artifacts.download_artifacts(run_id=row.run_id,artifact_path='pipeline/MLmodel'))
    text=model_path.read_text(encoding='utf-8')
    assert 'signature:' in text and 'saved_input_example_info:' in text
dump(DOCS/'tracking_audit.json',{'run_counts':counts,'total':len(rows),'all_finished':True,
    'all_90_trials_have_parent':True,'configurations_per_parent':30,
    'fitted_runs_have_models':True,'notes':'Three parent runs group child models and are not fitted estimators.'})
print(counts)
