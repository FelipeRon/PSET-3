"""Califica las respuestas guardadas de ChatGPT y registra la prueba en MLflow."""
import json
import math
from pathlib import Path

import mlflow
import pandas as pd

from src.datos import DOCS, dump, split
from src.experimentos import export_runs, tracking
from src.metricas import evaluate

folder = DOCS / "llm"
meta = json.loads((DOCS / "dataset.json").read_text(encoding="utf-8"))
key = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
if key["snapshot"] != meta["dataset_sha256"]:
    raise ValueError("La clave corresponde a otro snapshot")

rows = []
prefix = ("Clasifica falla actual en lecturas del conjunto sintético público AI4I, sin herramientas ni búsquedas. "
          "Estima cada probabilidad por separado. Orden de columnas: tipo, temperatura aire K, temperatura proceso K, "
          "rpm, torque Nm, desgaste min. En ejemplos el último valor es etiqueta (0 sin falla, 1 con falla). "
          "En consultas el primer valor es el identificador. Devuelve únicamente un objeto JSON con los 20 identificadores "
          "y su probabilidad entre 0 y 1.\n")
for fold in (1, 2, 3):
    response = json.loads((folder / f"response_{fold}.json").read_text(encoding="utf-8"))
    selected = [row for row in key["records"] if row["fold"] == fold]
    if set(response) != {row["id"] for row in selected} or len(selected) != 20:
        raise ValueError(f"Identificadores inválidos en fold {fold}")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in response.values()):
        raise ValueError(f"Probabilidad inválida en fold {fold}")
    rows.extend({**row, "probability": response[row["id"]]} for row in selected)
    prompt = (folder / f"prompt_{fold}.txt").read_text(encoding="utf-8").splitlines()
    examples, queries = [json.loads(line.split(": ", 1)[1]) for line in prompt[1:]]
    compact = lambda row: ",".join(format(col, "g") if isinstance(col, (int, float)) else col for col in row)
    sent = prefix + "Ejemplos: " + "; ".join(map(compact, examples)) + ".\nConsultas:\n" + "\n".join(map(compact, queries))
    (folder / f"submitted_{fold}.txt").write_text(sent, encoding="utf-8")

frame = pd.DataFrame(rows)
train, holdout, _ = split()
assert frame.reading_id.is_unique and set(frame.reading_id) <= set(train.reading_id)
assert set(frame.reading_id).isdisjoint(holdout.reading_id)
assert (frame.y == frame.reading_id.map(train.set_index("reading_id").machine_failure)).all()
frame.to_csv(DOCS / "llm_predictions.csv", index=False)
metrics = evaluate(frame.y, frame.probability, .5)
summary = {"status": "executed", "model": "ChatGPT web (versión interna no visible)", "date": "2026-11-20",
           "rows": len(frame), "positive_rows": int(frame.y.sum()), "threshold": .5,
           "metrics": metrics, "dataset_sha256": meta["dataset_sha256"],
           "protocol": "3 chats independientes. 6 ejemplos etiquetados y 20 consultas ciegas por fold. Solo train.",
           "scope": "Prueba exploratoria. La muestra pequeña no interviene en la selección del champion."}
tracking()
with mlflow.start_run(run_name="chatgpt_web_prompting") as run:
    mlflow.set_tags({"phase": "llm", "family": "chatgpt_web", "git_commit": "unavailable"})
    mlflow.log_params({"model_interface": "ChatGPT web", "seed": 42, "folds": 3, "threshold": .5,
                       "dataset_sha256": meta["dataset_sha256"], "source_table": meta["table"]})
    mlflow.log_metrics(metrics)
    summary["run_id"] = run.info.run_id
    dump(DOCS / "llm.json", summary)
    mlflow.log_artifacts(str(folder), "prompting")
    mlflow.log_artifact(str(DOCS / "llm_predictions.csv"))
    mlflow.log_artifact(str(DOCS / "llm.json"))
export_runs()
print(json.dumps(summary, ensure_ascii=False))
