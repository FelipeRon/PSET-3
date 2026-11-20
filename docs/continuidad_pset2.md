# Continuidad con el PSet 2

Código de referencia: https://github.com/FelipeRon/Pset-02, commit `6ebc989edeaa0a60fa408361b6832ad92438c6ab`.

Se conserva el contrato de spark/build_obt.py: una fila por reading_id del snapshot activo; sensores, tipo, diferencia térmica, potencia y machine_failure. Se heredan la regla desgaste >=200 o torque >=60 y el escenario Cp=500, Ce=8000, Cr=1500. Las métricas descriptivas del PSet 2 sobre todo el dataset no se presentan como resultados de test del PSet 3.

La corrida reproducida consulta PSET2.OBT.OBT_READINGS desde Snowflake. docs/dataset.json conserva el Query ID, la fecha técnica y el SHA-256. El respaldo corresponde a esa misma extracción.

UDI, Product ID y hash no son predictores. Los modos TWF/HDF/PWF/OSF/RNF describen el target y causarían fuga; quality_tier duplica tipo y baseline_alert es auditoría. Las features físicas se recalculan dentro del pipeline para inferir con sensores base.

PSet 3 utiliza su propio Compose con MLflow y no modifica la infraestructura PSet 2 ni copia sus contraseñas.
