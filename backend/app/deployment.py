"""Read measured deployment artifacts and apply operational constraints without promotion."""
import json
from pathlib import Path
from fastapi import APIRouter
from recipetriage_ml.alignment.preferences import ml_directory
from recipetriage_ml.deployment.benchmark import Constraints,select_candidate

router=APIRouter(prefix='/api/v1/deployment',tags=['Deployment measurements'])


def measured():
    root=ml_directory()/'deployment/results/v1'
    return [json.loads(p.read_text()) for p in sorted(root.glob('*.json')) if p.name not in {'comparison.json'}]


@router.get('/results')
def results():
    rows=measured();fields=['name','format','size_mib','peak_ram_mib','vram_mib','device','load_seconds','mean_ms','p95_ms','micro_f1','macro_f1','schema_validity','json_validity','measurement_scope']
    return {'candidates':[{k:row[k] for k in fields} for row in rows], 'constraints':Constraints().model_dump(),'selection':select_candidate(rows)}


@router.post('/select')
def select(constraints:Constraints):return select_candidate(measured(),constraints)
