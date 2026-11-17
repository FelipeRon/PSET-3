"""Construye notebooks limpios que importan src en vez de duplicar preparación."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
START="""from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from IPython.display import display, Image
import pandas as pd
from src.datos import DOCS, split
from src.pipeline import make_pipeline, PhysicalFeatures
"""
def save(name,cells):
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(cells[0]),nbf.v4.new_code_cell(START)]+[
        (nbf.v4.new_markdown_cell(x[1:]) if x.startswith('@') else nbf.v4.new_code_cell(x)) for x in cells[1:]])
    nb.metadata.kernelspec={'display_name':'Python PSet 3','language':'python','name':'pset3'}
    path=ROOT/'notebooks'/name;path.parent.mkdir(exist_ok=True);nbf.write(nb,path)

save('01_eda.ipynb',[
 '# EDA de la OBT\nSe inspecciona solo entrenamiento tras reservar 20 % estratificado. La procedencia ejecutada aparece debajo; UCI replay no se presenta como consulta Snowflake.',
 'from src.eda import run\nresult = run()\ndisplay(result)',
 "display(pd.read_csv(DOCS/'tipos_faltantes.csv'))\ndisplay(pd.read_csv(DOCS/'descriptivos.csv'))",
 "display(pd.read_csv(DOCS/'atipicos.csv'))\ndisplay(Image(filename=str(DOCS/'figuras/boxplots.png')))",
 "display(Image(filename=str(DOCS/'figuras/distribuciones.png')))\ndisplay(Image(filename=str(DOCS/'figuras/target.png')))",
 "display(pd.read_csv(DOCS/'spearman.csv'))\ndisplay(Image(filename=str(DOCS/'figuras/correlacion.png')))",
 '@Spearman se usa por colas e interés en relaciones monótonas sin imponer normalidad. Correlación con target binario es descriptiva, no causal. Los atípicos se conservan, pues pueden representar fallas. El desbalance orienta AP y CV estratificada; no se sobremuestrea antes de dividir. Tres hallazgos con consecuencia están en eda.json.'
])
save('02_features.ipynb',[
 '# Features disponibles en la lectura actual\nDiferencia térmica y potencia se recalculan dentro del pipeline. No se incorporan UDI, Product ID, subtipos de falla, baseline_alert, target ni hash.',
 "train, _, meta = split()\ndisplay(PhysicalFeatures().transform(train).head())\ncorr=pd.read_csv(DOCS/'spearman.csv',index_col=0)\ndisplay(corr[['machine_failure']])",
 '@La EDA muestra acoplamiento entre magnitudes físicas y una asociación de las features con el target. Diferencia térmica = proceso menos aire; potencia = torque por rpm por 2 pi / 60. Ambas existen en inferencia porque se calculan con sensores de la lectura, sin consultar la falla. El tipo se codifica one hot dentro de ColumnTransformer. La asociación no basta para afirmar utilidad; la ablación se ejecuta en el notebook 03, después de las líneas base.',
 "from sklearn.linear_model import LogisticRegression\npipe = make_pipeline(LogisticRegression(random_state=42))\ndisplay(pipe)",
 '@Mediana, moda y escalado se ajustan dentro de cada fold. La lista blanca ignora las columnas de auditoría y etiquetas. quality_tier duplica el tipo y se excluye. El proyecto detecta falla actual; no dispone de labels futuras ni historial por máquina.'
])
save('03_baseline_candidatos.ipynb',[
 '# Dos líneas base y cinco familias\nSe entrenan primero DummyClassifier y la regla desgaste >= 200 o torque >= 60; luego logística, Random Forest, HistGradientBoosting, MLP y GaussianNB con valores por defecto salvo semilla. Misma preparación y tres folds para todos.',
 'from src.experimentos import baseline_candidates\nrows = baseline_candidates()\ndisplay(pd.DataFrame(rows).drop(columns=["fold_ap","params"]))',
 "ablation=json.loads((DOCS/'ablacion.json').read_text(encoding='utf-8'))\ndefault=next(r for r in rows if r['name']=='hist_boosting')\ndisplay({'AP_con_features':default['cv_ap_mean'],'AP_sin_features':ablation['cv_ap_mean'], 'delta':default['cv_ap_mean']-ablation['cv_ap_mean'],'std_con_features':default['cv_ap_std']})",
 '@La brecha train AP menos CV AP ayuda a diagnosticar sobreajuste. La regla de negocio es una hipótesis del PSet anterior, pendiente de validación de un técnico. Las corridas se recuperan del checkpoint al reejecutar el notebook, sin duplicar búsqueda ni consumir el holdout.',
 'from src.llm import run as prompting\ndisplay(prompting())',
 '@El problema tabular puede formularse mediante prompting; se entrega un adaptador real para ChatGPT o Claude. Si no hay API key y modelo configurados, la salida indica not_executed y no afirma una evaluación inexistente. La prueba usa solo submuestras de folds de entrenamiento y nunca el holdout. Su presupuesto distinto no altera el ranking principal.'
])
save('04_tuning.ipynb',[
 '# Tuning de las tres mejores familias\n30 configuraciones por familia y la misma CV fija. ParameterSampler cubre espacios mixtos con presupuesto fijo; tasas, C y regularización se muestrean en log, tamaños en rangos discretos.',
 'from src.experimentos import tuning\ntrials=tuning()\ndisplay(pd.DataFrame(trials).drop(columns=["fold_ap"]).sort_values("cv_ap_mean",ascending=False))',
 "display(json.loads((DOCS/'plan_tuning.json').read_text(encoding='utf-8')))\nfrom src.experimentos import spaces\nprint(spaces())",
 "defaults=json.loads((DOCS/'candidatos.json').read_text(encoding='utf-8'))\ncomparison=[]\nfor family in sorted({r['name'].split('/')[0] for r in trials}):\n    initial=next(r for r in defaults if r['name']==family)\n    best=max((r for r in trials if r['name'].split('/')[0]==family),key=lambda r:r['cv_ap_mean'])\n    delta=best['cv_ap_mean']-initial['cv_ap_mean']\n    comparison.append({'family':family,'default_ap':initial['cv_ap_mean'],'tuned_ap':best['cv_ap_mean'],'delta':delta,'fold_std':best['cv_ap_std'],'mejora_supera_std':delta>max(initial['cv_ap_std'],best['cv_ap_std'])})\ndisplay(pd.DataFrame(comparison))",
 '@Una diferencia menor que la desviación entre folds no se interpreta como mejora demostrada. La repetición de selección y tuning en los mismos folds puede producir optimismo; no equivale a CV anidada. Las corridas hijas registran parámetros, métricas, warnings y pipeline completo con firma.'
])
save('05_seleccion.ipynb',[
 '# Selección congelada y reproducción desde Model Registry\nEl ganador y challenger se eligen por CV antes de abrir el holdout. Se evalúan una sola vez; el resultado de prueba no cambia la elección. Las reejecuciones leen la decisión guardada y verifican el alias, sin elegir de nuevo.',
 'from src.experimentos import final, reproduce, export_runs\nselection=final()\ndisplay(pd.DataFrame([{**{k:r[k] for k in ["alias","name","final_run_id","version","threshold","cv_ap_mean","cv_ap_std"]}, **r["test"]} for r in selection["models"]]))\ndisplay({"cv_difference":selection["cv_difference"],"exceeds_fold_std":selection["difference_exceeds_fold_std"]})',
 '@El costo simulado usa Cp=500, Ce=8000 y Cr=1500 USD, heredados del PSet 2 como supuestos. Se elige umbral con predicciones out of fold del entrenamiento y máximo 12 % de alertas en ese conjunto. La tasa en holdout se reporta; no se garantiza capacidad futura. AP no depende de umbral. Las probabilidades no se declaran calibradas.',
 'verification=reproduce()\ndisplay(verification)\nassert verification["reproduced"]',
 'exported=export_runs()\ndisplay(exported.groupby("tags.phase").size())\nprint("docs/corridas.csv actualizado con mlflow.search_runs")',
 '@La reproducción en kernel limpio carga models:/ai4i-mantenimiento-modelo@champion y exige igualdad de métricas con tolerancia 1e-12. Los datos sintéticos no demuestran reducción de paradas ni transferencia a planta. El registro deja un candidato reproducible para PSet 4, no una autorización de despliegue industrial.'
])
print('Cinco notebooks creados')
