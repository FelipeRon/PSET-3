"""Prepare three independent, blinded ChatGPT prompts from the training split."""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.datos import BASE, SEED, TARGET, split


def main():
    train, _, meta = split()
    x, y = train[BASE], train[TARGET].to_numpy()
    prompts, key = [], []
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(cv.split(x, y), 1):
        rng = np.random.default_rng(SEED + fold - 1)
        examples = []
        for label in (0, 1):
            for index in rng.choice(a[y[a] == label], 3, replace=False):
                examples.append([*x.iloc[index].tolist(), label])
        queries = []
        for number, index in enumerate(rng.choice(b, 20, replace=False), 1):
            token = f"{fold}-{number:02d}"
            queries.append([token, *x.iloc[index].tolist()])
            key.append({"id": token, "reading_id": int(train.iloc[index].reading_id),
                        "fold": fold, "y": int(y[index])})
        prompt = (
            "Clasifica falla actual en cada lectura del conjunto sintético público AI4I. "
            "Usa solo estas mediciones y los ejemplos etiquetados. Estima cada probabilidad "
            "por separado. No busques el dataset, no uses herramientas y no infieras fallas futuras. "
            "Las columnas son tipo, temperatura aire K, temperatura proceso K, velocidad rpm, "
            "torque Nm y desgaste min. En los ejemplos la última columna es la etiqueta "
            "(0 sin falla, 1 con falla). Cada consulta empieza con un identificador arbitrario. "
            'Devuelve solo un objeto JSON que mapee los 20 identificadores a probabilidades entre 0 y 1.\n'
            f"Ejemplos: {json.dumps(examples)}\nConsultas: {json.dumps(queries)}"
        )
        prompts.append(prompt)
    output = Path("docs/llm")
    output.mkdir(exist_ok=True)
    for fold, prompt in enumerate(prompts, 1):
        (output / f"prompt_{fold}.txt").write_text(prompt, encoding="utf-8")
    (output / "labels.json").write_text(json.dumps({"snapshot": meta["dataset_sha256"],
                                                  "records": key}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
