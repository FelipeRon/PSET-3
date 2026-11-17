"""Ejecuta cada notebook con un kernel nuevo y guarda todas sus salidas."""
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'artifacts/jupyter/kernels/pset3';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Python PSet3','language':'python'}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(ROOT/'artifacts/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    NotebookClient(nb,timeout=7200,kernel_name='pset3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path);print(path.name+' ejecutado',flush=True)
