"""Export a checked unquantized LoRA adapter as a standalone safetensors model."""
import argparse,json
from pathlib import Path
from recipetriage_ml.training.sft import load_adapter
from recipetriage_ml.training.data import file_hash
from recipetriage_ml.data.pipeline import digest


def merge(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Choose a new model version directory; never overwrite weights')
    identity=json.loads((source/'adapter-provenance.json').read_text())
    if identity.get('quantization') not in {None,'none'}:
        raise ValueError('Merge the full-precision LoRA path. NF4 adapter merging needs explicit dequantization/revalidation and is not silently substituted here.')
    provider=load_adapter(source)
    merged=provider.model.merge_and_unload(safe_merge=True);merged.eval()
    merged.save_pretrained(output,safe_serialization=True,max_shard_size='2GB')
    provider.tokenizer.save_pretrained(output)
    for path in output.iterdir():
        if path.is_file():path.chmod(0o644)  # read-only runtime bind mounts use a different UID
    artifact={'format':'hf-safetensors-fp32','source_adapter':identity,'files':{p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
              'dtype':'float32','merged':True,'production_changed':False}
    artifact['artifact_sha256']=digest(artifact['files'])
    (output/'deployment-manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    return artifact


def export_base(output):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from recipetriage_ml.evaluation.base_provider import MODEL_ID,MODEL_REVISION
    output=Path(output)
    if output.exists():raise ValueError('Choose a new base export directory')
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,dtype=torch.float32,trust_remote_code=False)
    model.save_pretrained(output,safe_serialization=True,max_shard_size='2GB')
    AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION).save_pretrained(output)
    for path in output.iterdir():
        if path.is_file():path.chmod(0o644)
    artifact={'format':'hf-safetensors-fp32','source_adapter':None,'base_model':MODEL_ID,'base_revision':MODEL_REVISION,
              'files':{p.name:file_hash(p) for p in output.iterdir() if p.is_file()},'dtype':'float32','merged':False,'production_changed':False}
    artifact['artifact_sha256']=digest(artifact['files']);(output/'deployment-manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    return artifact


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',type=Path);parser.add_argument('--base',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if bool(args.adapter)==args.base:parser.error('Choose --adapter PATH or --base')
    print(json.dumps(export_base(args.output) if args.base else merge(args.adapter,args.output),indent=2))
