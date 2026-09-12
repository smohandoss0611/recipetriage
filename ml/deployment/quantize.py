"""Supported portable CPU format: bitsandbytes NF4 on a merged HF safetensors model."""
import argparse,json
from pathlib import Path
from recipetriage_ml.training.qlora import quantization_config,capability
from recipetriage_ml.training.data import file_hash
from recipetriage_ml.data.pipeline import digest


def quantize(source,output):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Choose a new quantized version directory')
    if not capability()['supported']:raise ValueError(capability()['error'])
    manifest=json.loads((source/'deployment-manifest.json').read_text())
    if manifest['format']!='hf-safetensors-fp32':raise ValueError('Quantization input must be the checked merged float32 export')
    for name,h in manifest['files'].items():
        if file_hash(source/name)!=h:raise ValueError('Merged source checksum changed')
    model=AutoModelForCausalLM.from_pretrained(source,quantization_config=quantization_config(),device_map={'':'cpu'},dtype=torch.float32,trust_remote_code=False)
    import bitsandbytes as bnb
    count=sum(isinstance(m,bnb.nn.Linear4bit) and m.weight.quant_state.quant_type=='nf4' for m in model.modules())
    if count!=168:raise ValueError('NF4 deployment conversion did not cover all 168 inspected projections')
    model.save_pretrained(output,safe_serialization=True);AutoTokenizer.from_pretrained(source).save_pretrained(output)
    for path in output.iterdir():
        if path.is_file():path.chmod(0o644)
    result={'format':'hf-bitsandbytes-nf4','source_sha256':manifest['artifact_sha256'],'source_adapter':manifest['source_adapter'],
        'quantization':quantization_config().to_dict(),'quantized_linear_modules':count,
        'files':{p.name:file_hash(p) for p in output.iterdir() if p.is_file()},'production_changed':False}
    result['artifact_sha256']=digest(result['files']);(output/'deployment-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(quantize(args.source,args.output),indent=2))
