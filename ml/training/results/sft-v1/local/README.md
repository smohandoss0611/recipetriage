# Local SFT reference evidence

This is a completed CPU LoRA run on the five-recipe training split. `run.json` contains the full configuration, history, dataset hashes, before/after benchmark responses and measured comparison. `adapter/` holds the selected adapter and tokenizer; load it with `recipetriage_ml.training.sft.load_adapter(this_directory)` to verify its checksum and pinned base. The original trainer checkpoints remain in the originating run directory and are not bundled.

Micro-F1 fell from 48.48% to 38.71%; JSON validity rose from 50% to 90%. No generalization improvement is claimed. `console.log` includes the actual library warnings.
