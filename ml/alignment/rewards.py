"""Transparent bounded reward; a label-policy proxy, not a human preference model."""
from recipetriage_ml.evaluation.metrics import inspect_response,prf

WEIGHTS={'label_f1':.55,'json_valid':.10,'schema_valid':.10,'uncertainty':.15,'shortcut_consistency':.10}
VERSION='recipe-reward-v1'


def components(raw,expected_labels,*,counterfactual_raw=None,finish_reason='stop',counterfactual_finish='stop'):
    checked=inspect_response(raw,finish_reason);predicted=set(checked['prediction']['labels']) if checked['usable'] else set();truth=set(expected_labels)
    label_f1=prf(len(predicted&truth),len(predicted-truth),len(truth-predicted))['f1'] if checked['usable'] else 0.
    uncertainty=float(checked['usable'] and (('unclear' in predicted)==('unclear' in truth)))
    robust=0.
    if counterfactual_raw is not None:
        other=inspect_response(counterfactual_raw,counterfactual_finish)
        # Consistently wrong/invalid answers never receive the robustness bonus.
        robust=float(checked['usable'] and other['usable'] and predicted==truth==set(other['prediction']['labels']))
    values={'label_f1':label_f1,'json_valid':float(checked['json_valid']),'schema_valid':float(checked['usable']),
            'uncertainty':uncertainty,'shortcut_consistency':robust}
    return {**values,'total':sum(WEIGHTS[k]*v for k,v in values.items()),'counterfactual_measured':counterfactual_raw is not None}


def make_grpo_reward(log, counterfactual=None, eos_token_id=None):
    def recipe_reward(prompts,completions,expected_labels,**kwargs):
        values=[];paired={}
        for index,(prompt,raw,labels) in enumerate(zip(prompts,completions,expected_labels)):
            text=raw if isinstance(raw,str) else raw[0]['content']
            if counterfactual and prompt not in paired:paired[prompt]=counterfactual(prompt)
            other=paired.get(prompt)
            ids=kwargs.get('completion_ids')
            finish='stop' if eos_token_id is None or ids and ids[index] and ids[index][-1]==eos_token_id else 'length'
            score=components(text,labels,finish_reason=finish,counterfactual_raw=other.raw_output if other else None,
                             counterfactual_finish=other.finish_reason if other else 'stop')
            log.append({'prompt':prompt,'completion':text,'expected_labels':labels,'components':score,
                        'counterfactual':other.model_dump() if other else None,'finish_reason':finish})
            values.append(score['total'])
        return values
    return recipe_reward
