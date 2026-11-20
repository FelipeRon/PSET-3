"""Gráficos y memo de seis páginas principales más anexos medidos."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from src.datos import DOCS, BASE

def read(name):return json.loads((DOCS/name).read_text(encoding='utf-8'))
candidates=read('candidatos.json');trials=read('tuning.json');selection=read('seleccion.json');eda=read('eda.json');meta=read('dataset.json')
figures=DOCS/'figuras';figures.mkdir(exist_ok=True)
rows=pd.DataFrame(candidates)
fig,ax=plt.subplots(figsize=(8,3))
x=np.arange(len(rows));ax.bar(x-.18,rows.train_ap,.36,label='Train AP',color='#8198AC')
ax.bar(x+.18,rows.cv_ap_mean,.36,yerr=rows.cv_ap_std,label='CV AP',color='#245C79',capsize=3)
ax.set_xticks(x,rows.name,rotation=25,ha='right',fontsize=8);ax.legend();ax.set_ylim(0,1.05);fig.tight_layout()
fig.savefig(figures/'candidatos.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3))
for family in sorted({r['name'].split('/')[0] for r in trials}):
    arr=[r for r in trials if r['name'].startswith(family+'/')]
    ax.plot(range(1,31),[r['cv_ap_mean'] for r in arr],'.-',label=family)
ax.set_xlabel('Configuración aleatoria');ax.set_ylabel('AP de CV');ax.legend();fig.tight_layout()
fig.savefig(figures/'tuning.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8,3))
for ax,r in zip(axes,selection['models']):
    m=r['test'];a=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    ax.imshow(a,cmap='Blues');ax.set_title(r['alias']);ax.set_xticks([0,1],['Pred 0','Pred 1']);ax.set_yticks([0,1],['Real 0','Real 1'])
    for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>a.max()/2 else 'black')
fig.tight_layout();fig.savefig(figures/'confusion.png',dpi=160);plt.close(fig)

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.3,leading=12.5,spaceAfter=7))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=7.5,leading=10,spaceAfter=5))
for n in ['Title','Heading1','Heading2']:styles[n].textColor=colors.black
styles['Title'].fontSize=20;styles['Title'].leading=24
styles['Heading1'].fontSize=15;styles['Heading1'].leading=18
styles['Heading2'].fontSize=11;styles['Heading2'].leading=14
story=[]
def p(text,small=False):story.append(Paragraph(text,styles['SmallCustom' if small else 'BodyCustom']))
def h(text):story.append(Paragraph(text,styles['Heading2']))
def page(title):story.append(PageBreak());story.append(Paragraph(title,styles['Heading1']))
def img(name,w=460,h=175):story.append(Image(str(figures/name),width=w,height=h));story.append(Spacer(1,8))
def table(headers,data,widths):
    entries=[[Paragraph(str(c),styles['SmallCustom']) for c in row] for row in [headers]+data]
    t=Table(entries,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DAE5ED')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F5F7')]),
        ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t);story.append(Spacer(1,9))
f=lambda n: f'{n:.4f}'
champ,chall=selection['models']
story.append(Paragraph('Experimentación de falla actual con AI4I',styles['Title']))
p('PSet 3 | Felipe Ron Andrade | 20 de noviembre de 2026')
h('1 Evidencia y exploración')
p(f'La selección por CV favorece <b>{champ["name"]}</b>, registrado como champion. Su AP de CV es {f(champ["cv_ap_mean"])} +/- {f(champ["cv_ap_std"])} y su AP de prueba es {f(champ["test"]["ap"])}. La diferencia de CV frente al segundo {"supera" if selection["difference_exceeds_fold_std"] else "no supera"} la variación entre folds; se reporta esa incertidumbre sin convertir el ranking en superioridad concluyente.')
p(('La corrida incluida reconstruye el contrato OBT del PSet 2 desde UCI oficial. No es una consulta Snowflake. La lectura directa de PSET2.OBT.OBT_READINGS está implementada para repetir el protocolo en la cuenta del autor; esta sesión no utiliza su llave privada. ' if meta['source']=='uci_replay' else 'La corrida usa una exportación autenticada de '+meta['table']+'; Query id y fecha técnica están en dataset.json. ')+'Se detecta falla actual en datos sintéticos, sin atribuirles historia por máquina ni RUL.')
p(f'El snapshot tiene {meta["rows"]:,} lecturas y {meta["target_failures"]} fallas. Se reservó 20 % antes de EDA; los hallazgos se calculan en {eda["train_rows"]:,} filas de entrenamiento, con {eda["prevalence_train"]:.2%} de positivos. Faltantes, tipos, descriptivos y atípicos completos constan en anexos CSV.')
table(['Hallazgo','Evidencia train','Decisión'],[
 ['Desbalance',f'{eda["positives_train"]} positivos de {eda["train_rows"]}','Usar AP y folds estratificados; accuracy no elige.'],
 ['Colas físicas','IQR y |Z| >= 3 en docs/atipicos.csv','Conservar extremos y comparar familias no lineales.'],
 ['Sensores acoplados',f'Spearman delta térmica: {f(eda["feature_target_spearman"]["temperature_delta_k"])}; potencia: {f(eda["feature_target_spearman"]["power_w"])}','Derivar features físicas y contrastar aporte con ablación.']],[90,175,215])
img('correlacion.png',330,235)
p('Spearman describe asociación monótona con colas y sin suponer normalidad; no implica causalidad. No se inspecciona test para diseñar features.',True)

page('2 Features y preparación sin fuga')
p('La diferencia térmica es proceso menos aire. La potencia es torque por rpm por 2 pi / 60. Ambas se calculan con magnitudes de la lectura, disponibles en inferencia. PhysicalFeatures las recalcula en cada ejecución; no usa labels, estadísticas del holdout ni reglas del generador de fallas.')
table(['Feature','Evidencia y disponibilidad','Tratamiento'],[
 ['temperature_delta_k','La EDA train mide su asociación; ambas temperaturas existen en la lectura.','Numérica; imputación mediana y escalado dentro del fold.'],
 ['power_w','La EDA train relaciona combinación de torque y velocidad; unidades físicas de potencia.','Numérica; comparación posterior con y sin derivadas.'],
 ['product_type','Categoría disponible antes de clasificar; L M H.','Moda y one hot con categorías desconocidas toleradas.']],[105,215,160])
ablation=read('ablacion.json');hb=next(r for r in candidates if r['name']=='hist_boosting')
p(f'Ablación controlada, misma CV y boosting default: AP con derivadas {f(hb["cv_ap_mean"])}, sin derivadas {f(ablation["cv_ap_mean"])}, delta {f(hb["cv_ap_mean"]-ablation["cv_ap_mean"])}. La magnitud se compara con desviaciones {f(hb["cv_ap_std"])} y {f(ablation["cv_ap_std"])}; no se asume aporte solo por plausibilidad física.')
img('boxplots.png',480,195)
h('División y exclusiones')
p('El split estratificado 80/20 con semilla 42 se persiste por UDI en split.json. Los tres folds estratificados de train se reutilizan en todas las fases. Sin fechas ni grupos de activos acreditados, no se inventa CV temporal o por máquina. El pipeline compartido contiene ColumnTransformer, imputación, codificación y escalado; cada fit de CV aprende solo de su train.')
p('La lista blanca excluye machine_failure de las entradas y elimina UDI, Product ID, hash, baseline_alert, quality_tier y modos TWF/HDF/PWF/OSF/RNF. Estos últimos describen la falla y causarían leakage. La política se prueba automáticamente. No se recortan outliers globalmente ni se sobremuestrea antes de dividir.')
h('Métricas comunes')
p('Average Precision es la métrica técnica para todos. Se conserva recall como diagnóstico relevante del PSet 2. El costo por 1.000 lecturas pondera alertas, falsos negativos y daño residual con Cp=500, Ce=8.000, Cr=1.500 USD. Son hipótesis económicas, no costos medidos ni ahorro demostrado.')

page('3 Diseño experimental y candidatos')
p('Antes de cualquier candidato se ejecutaron DummyClassifier de clase mayoritaria y la regla de negocio desgaste >=200 o torque >=60. Después se entrenaron cinco familias con defaults salvo semilla, la misma preparación y los mismos folds. Cada corrida del protocolo principal guarda parámetros, semilla, dataset, corte técnico, hash, fase, familia, métricas, tiempo y pipeline con firma y ejemplo. La copia sin historial Git registra commit unavailable. Se guardan advertencias de convergencia, no se ocultan.')
table(['Modelo','Train AP','CV AP +/- std','Brecha','Fit s','LR'],[
 [r['name'],f(r['train_ap']),f(r['cv_ap_mean'])+' +/- '+f(r['cv_ap_std']),f(r['gap_ap']),f'{r["fit_seconds"]:.2f}',str(r['learning_rate'] or 'n/a')] for r in candidates],[105,55,115,60,60,85])
img('candidatos.png',480,200)
p('La brecha compara el refit en todo train con el promedio de CV; no es una prueba estadística. Las AP de train altas en árboles pueden señalar sobreajuste. Los resultados de folds y warnings están en artefactos MLflow. La regla produce score binario; su AP no describe una curva probabilística fina.')
baselines=[r for r in candidates if r['phase']=='baseline']
failed=[r['name'] for r in candidates if r['phase']=='candidato' and (r['cv_ap_mean']<=max(b['cv_ap_mean'] for b in baselines) or r['cv_cost_per_1000']>=min(b['cv_cost_per_1000'] for b in baselines))]
p('Comparación de ambas líneas base: todos los candidatos se contrastan con AP y costo de CV a umbral 0,5. Modelos que no superan simultáneamente ambos criterios: '+(', '.join(failed) or 'ninguno')+'. Una AP mayor no garantiza mejor costo al mismo umbral; el umbral final se decide solo en train OOF.')
p('El presupuesto principal es 2 baselines + 5 candidatos + 90 configuraciones + 2 finalistas. Se añade una ablación y tres padres de tuning. El CSV exportado permite auditar el conteo exacto. La captura real de la comparación en MLflow consta en el anexo de evidencia.')

page('4 Tuning y prueba de prompting')
p('Las tres mejores familias por AP inicial recibieron 30 configuraciones cada una mediante ParameterSampler con semilla 42. Se eligió búsqueda aleatoria por presupuesto fijo y espacios mixtos. Las mismas particiones impiden que una diferencia de split explique la mejora. Cada configuración es una corrida hija con su pipeline completo, dentro de un padre por familia.')
table(['Familia','Rangos de búsqueda y escala'],[
 ['Boosting','learning_rate log 0,02 a 0,3; iteraciones 50 a 150; hojas 7 a 31; L2 log 0,0001 a 10; hoja mínima 10 a 49.'],
 ['Random Forest','50 a 150 árboles; profundidad None o 4 6 10 16; hoja mínima 1 a 11; max_features sqrt o 0,7 o 1; class_weight None o balanced.'],
 ['MLP','Capas 32 o 64 o 100 o 32 16; alpha log 0,00001 a 0,1; tasa inicial log 0,0001 a 0,01; max_iter 200 o 300.']],[100,380])
comparison=[]
for family in read('plan_tuning.json')['selected_families']:
    initial=next(r for r in candidates if r['name']==family)
    best=max((r for r in trials if r['name'].startswith(family+'/')),key=lambda r:r['cv_ap_mean'])
    delta=best['cv_ap_mean']-initial['cv_ap_mean']
    comparison.append([family,f(initial['cv_ap_mean']),f(best['cv_ap_mean']),f(delta),f(best['cv_ap_std']),'Sí' if delta>max(initial['cv_ap_std'],best['cv_ap_std']) else 'No'])
table(['Familia','Default','Tuned','Delta','Std tuned','Supera std'],comparison,[100,65,65,65,75,110])
img('tuning.png',480,180)
p('Una mejora menor que la dispersión entre folds no se interpreta como ganancia demostrada. La mejor configuración puede ser optimista porque búsqueda y ranking reutilizan CV; no se afirma CV anidada.')
llm=read('llm.json') if (DOCS/'llm.json').exists() else {'status':'not_executed'}
h('ChatGPT o Claude')
p('Estado de la prueba LLM: <b>'+llm['status']+'</b>. El problema numérico sí admite prompting. Se implementó un adaptador para ambos proveedores, con prompt versionado, ejemplos solo del train del fold, respuesta JSON y 60 lecturas de validación interna; nunca se envía holdout. '+('Sin API key y modelo autorizados no se ejecutó la llamada y no hay una métrica LLM atribuida. Esta parte de la guía permanece pendiente.' if llm['status']!='executed' else 'Modelo utilizado: '+llm['model']+'. AP '+f(llm['metrics']['ap'])+' en '+str(llm['rows'])+' consultas. El anexo C documenta los tres chats ciegos y sus respuestas.')+' Su presupuesto distinto exige comparación cautelosa.')

page('5 Selección y Model Registry')
p('El ranking y los dos finalistas de familias distintas se congelaron por AP de CV antes de abrir test. El umbral se eligió minimizando costo sobre probabilidades out of fold de train, sujeto a máximo 12 % de alertas en ese conjunto. El holdout se evaluó una vez por finalista. Los resultados de test no cambiaron la selección.')
table(['Indicador','Champion','Challenger'],[
 ['Modelo',champ['name'],chall['name']],['CV AP +/- std',f(champ['cv_ap_mean'])+' +/- '+f(champ['cv_ap_std']),f(chall['cv_ap_mean'])+' +/- '+f(chall['cv_ap_std'])],
 ['Test AP',f(champ['test']['ap']),f(chall['test']['ap'])],['Umbral',f(champ['threshold']),f(chall['threshold'])],
 ['Recall test',f(champ['test']['recall']),f(chall['test']['recall'])],['Precisión test',f(champ['test']['precision']),f(chall['test']['precision'])],
 ['Costo test por 1.000 USD',f'{champ["test"]["cost_per_1000"]:,.2f}',f'{chall["test"]["cost_per_1000"]:,.2f}'],
 ['Alertas test',f'{champ["test"]["alert_rate"]:.2%}',f'{chall["test"]["alert_rate"]:.2%}'],
 ['Inferencia lote test s',f'{champ["inference_seconds"]:.4f}',f'{chall["inference_seconds"]:.4f}'],
 ['Versión registrada',champ['version'],chall['version']]],[180,150,150])
img('confusion.png',420,145)
p(f'La diferencia de CV es {f(selection["cv_difference"])} y {"supera" if selection["difference_exceeds_fold_std"] else "no supera"} la mayor desviación de los dos. La elección champion preserva el criterio previamente definido; si el segundo obtiene mejor test o costo, se documenta, sin reabrir test para optimizar el ranking.')
p('Registro: ai4i-mantenimiento-modelo. Aliases champion y challenger. Cada versión empaqueta sensores base, transformación, imputación, codificación, escalado, estimador y umbral. El notebook 05 carga models:/ai4i-mantenimiento-modelo@champion en kernel limpio y exige reproducción numérica de las métricas registradas.')
p('Run id champion: '+champ['final_run_id']+'<br/>Run id challenger: '+chall['final_run_id'],True)

page('6 Decisión limitaciones y reproducibilidad')
h('Decisión respaldada por evidencia')
p(f'Se conserva {champ["name"]} como champion por el ranking de AP de CV congelado. El segundo permanece challenger para futuras pruebas sin modificar la regla de selección con el holdout. La variación de folds no se confunde con intervalos de confianza; tres folds no acreditan superioridad estadística. Los archivos contienen errores y costos por ambos finalistas.')
if (DOCS/'importancias_train.csv').exists():
    imp=pd.read_csv(DOCS/'importancias_train.csv');top=imp[imp.alias=='champion'].nlargest(3,'ap_drop_mean')
    info='; '.join(f'{r.feature}: caída AP {r.ap_drop_mean:.3f}' for r in top.itertuples())
else:info='análisis de permutación todavía no ejecutado'
p('La explicabilidad se apoya en variables físicas y permutación de sensores en train. Champion: '+info+'. La muestra explicativa contiene positivos train y 1.000 negativos; su prevalencia enriquecida no representa evaluación. Sensores correlacionados comparten importancia; estos valores son descriptivos, no causales ni explicaciones individuales. El tiempo de inferencia medido sobre el lote no representa un SLA.')
h('Limitaciones materiales')
for text in [
 'AI4I es sintético y describe falla actual. No permite probar anticipación, RUL ni transferencia a una planta real. Hay discrepancias de etiquetas documentadas en PSet 2.',
 ('La corrida entregada reconstruye localmente el contrato OBT desde UCI. Falta repetir la experimentación con la exportación real de Snowflake para acreditar ese origen en el PSet 3.' if meta['source']=='uci_replay' else 'La extracción Snowflake fija una fecha técnica, no una fecha de observación industrial. El snapshot no aporta historia temporal por activo.'),
 'El tuning utiliza la misma CV para buscar y comparar. Las mejoras menores que std no se declaran demostradas. Se conservan las advertencias de convergencia de MLP.',
 'Cp, Ce, Cr y capacidad son supuestos. La efectividad de actuar sobre falla actual no fue validada. Las probabilidades no se declaran calibradas; el umbral empírico no garantiza capacidad futura.',
 ('La prueba real de LLM requiere credencial privada y modelo explícito. Sin esa llamada la exigencia de prompting no está cerrada. No se usa esta limitación para declarar imposible el problema.' if llm['status']!='executed' else 'La evaluación LLM usa una submuestra y no tiene el mismo presupuesto que la CV principal; no permite atribuirle superioridad con 60 lecturas.'),
 'La entrega registra un candidato reproducible para PSet 4. Despliegue industrial, monitoreo y piloto prospectivo requieren datos y validaciones adicionales.']:
 p('- '+text)
h('Reproducción y fuentes')
p('Cinco notebooks se ejecutan con kernel nuevo y salidas guardadas. src/pipeline.py concentra la preparación. El respaldo MLflow conserva datos, modelos y SQLite; src.portable restaura las rutas sin sobrescribir un tracking existente. docs/corridas.csv se genera con mlflow.search_runs. El README describe restauración, lectura Snowflake y orden de ejecución.')
p('Fuentes: guía CD_PSet_3 del autor; UCI AI4I DOI 10.24432/C5HS5C; contrato y costos del repositorio PSet 2; documentación oficial scikit-learn y MLflow. El SHA-256 de datos se verifica contra los artefactos. Los runs de esta copia registran git_commit=unavailable. El código y evidencia se entregan en https://github.com/FelipeRon/PSET-3.',True)

page('Anexo A Evidencia de MLflow y registro')
shot=DOCS/'evidencias/mlflow_snowflake.jpg'
if shot.exists():
    from PIL import Image as PILImage
    im=PILImage.open(shot);w,hp=im.size
    story.append(Image(str(shot),width=480,height=480*hp/w))
else:p('Captura de comparación pendiente de insertar. Los resultados tabulares permanecen disponibles en corridas.csv.')
p('Captura de la interfaz de MLflow para los dos finalistas de la corrida de Snowflake. Muestra los IDs, la AP media de CV, la desviación y la AP de prueba. La tabla de la página 5 y corridas.csv registran los valores con mayor precisión y el costo de negocio.')
table(['Alias','Versión','Run id'],[[r['alias'],r['version'],r['final_run_id']] for r in selection['models']],[90,60,330])
p('La métrica se reproduce cargando el alias champion desde un kernel limpio. El snapshot incluido permite repetir la comprobación sin acceder al servidor original. Para un experimento nuevo, las capturas deben corresponder a esa nueva corrida.')

page('Anexo B Métricas negocio y trazabilidad')
p('Métrica técnica: AP media de los tres folds. Dispersión: desviación muestral ddof=1. Brecha: AP de entrenamiento completo menos AP media de CV. Costo = (TP+FP) Cp + FN Ce + TP Cr; se normaliza por 1.000 lecturas. El target de cada fold no interviene en imputación o escalado.')
p('Snapshot UTC: '+meta['snapshot_utc']+'<br/>Origen ejecutado: '+meta['source']+'<br/>Tabla objetivo del contrato: '+meta['table'],True)
p('SHA 256 canónico: '+meta['dataset_sha256'],True)
if 'query_id' in meta:p('Query ID de Snowflake: '+meta['query_id'],True)
if 'csv_sha256' in meta:p('SHA 256 CSV UCI: '+meta['csv_sha256'],True)
table(['Archivo','Contenido'],[
 ['tipos_faltantes.csv y descriptivos.csv','Tipos, vacíos y estadísticos en train.'],['atipicos.csv','Conteos IQR y |Z| >= 3; límites por variable.'],
 ['spearman.csv y eda.json','Asociaciones y hallazgos con decisiones.'],['candidatos.json y tuning.json','AP, std, brecha, tiempo, warnings y run id.'],
 ['decision_before_holdout.json','Ranking y umbrales congelados antes de test.'],['seleccion.json','Métricas finales, aliases y versiones.'],
 ['corridas.csv','Exportación real desde mlflow.search_runs.'],['llm.json','Prueba realizada o motivo explícito de pendiente.']],[210,270])
p('Los datos de test se usan una vez para cerrar la comparación. El notebook de reproducción vuelve a inferir con el mismo modelo y snapshot únicamente para verificar el resultado; no habilita una nueva decisión de selección.')

page('Anexo C Evidencia de la prueba con ChatGPT')
if llm['status']=='executed':
    metric=llm['metrics']
    p('Tres chats independientes. Cada chat recibió seis ejemplos etiquetados de train de su fold y 20 lecturas ciegas de validación interna. La muestra conserva la asignación predefinida y excluye el holdout.')
    table(['Indicador','Valor'],[
        ['Interfaz',llm['model']],['Run ID',llm['run_id']],['Muestra',f'{llm["rows"]} lecturas, {llm["positive_rows"]} fallas'],
        ['Average Precision',f(metric['ap'])],['ROC AUC',f(metric['roc_auc'])],['Umbral exploratorio',llm['threshold']],
        ['TP / FP / FN / TN',f'{metric["tp"]} / {metric["fp"]} / {metric["fn"]} / {metric["tn"]}'],
        ['Costo simulado USD por 1.000',f'{metric["cost_per_1000"]:,.0f}']],[205,275])
    p('Los archivos docs/llm/submitted_1.txt a submitted_3.txt reproducen los mensajes. response_1.json a response_3.json conservan las salidas. labels.json vincula los identificadores ciegos con la verdad de terreno. llm_predictions.csv permite recalcular resultados y el run de MLflow contiene los artefactos.')
    p('Con el umbral exploratorio de 0,50, ChatGPT no emitió alertas y omitió las tres fallas. La muestra es exploratoria y su AP no es directamente comparable con la CV completa.')

page('Anexo D Reproducción en un entorno limpio')
if (DOCS/'evidencias/reproduccion_portable.json').exists():
    proof=read('evidencias/reproduccion_portable.json')
    p('Se aplicó el README desde un directorio recién creado, con un proyecto Compose y un servidor MLflow separados. La copia empezó sin data, artifacts ni llave de Snowflake. Se usó el respaldo de esta misma corrida.')
    table(['Comprobación','Evidencia'],[
        ['Respaldo de la corrida','docs/mlflow_snapshot.zip'],
        ['SHA-256 del ZIP',proof['archive_sha256']],
        ['Verificación de alias',str(proof['verification']['reproduced'])],
        ['Average Precision reproducida',f(proof['verification']['metrics']['ap'])],
        ['Costo simulado por 1.000',proof['verification']['metrics']['cost_per_1000']],
        ['Pruebas automáticas',proof['tests']],
        ['Entorno',proof['environment']]],[155,325])
    p('La evidencia íntegra se conserva en docs/evidencias/reproduccion_portable.json. La prueba valida la restauración y la inferencia en una copia aislada sobre el mismo equipo físico. Un entrenamiento de cero consulta Snowflake y requiere las credenciales y permisos del ejecutor.')
else:
    p('La prueba de restauración limpia se documenta al completar la verificación.')

page('Anexo E Distribuciones y fuentes de auditoría')
p('Las distribuciones se describen exclusivamente con el conjunto de entrenamiento. El desbalance justifica Average Precision y la estratificación.')
img('distribuciones.png',470,250)
img('target.png',330,155)
p('La tabla de descriptivos, los límites de detección de atípicos y la matriz de Spearman se incluyen en los CSV de docs. Los notebooks conservan las salidas y el CSV de corridas identifica cada evaluación.')

def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.drawRightString(552,23,f'Felipe Ron Andrade | PSet 3 | {doc.page}')

dest=DOCS/'PSet3_memo_Ron_Andrade.pdf'
SimpleDocTemplate(str(dest),pagesize=(595.28,841.89),leftMargin=52,rightMargin=52,
    topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
print(dest)
