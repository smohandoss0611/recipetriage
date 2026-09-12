"""Controlled title-only experiment; records successes, malformed outputs and blockers."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version as package_version, PackageNotFoundError
from importlib.resources import files
import json
import os
from pathlib import Path
import platform
from uuid import uuid4
from .contracts import Recipe, ProviderError
from .prompts import build_messages, PROMPT_VERSION
from .service import triage

TITLES = ["Easy", "Traditional", "Quick", "Simple"]


def fixture():
    return json.loads(files("recipetriage_ml").joinpath("examples/ravioli.json").read_text())


def variants(recipe):
    return [Recipe.model_validate({**recipe, "title": f"{title} Ravioli"}) for title in TITLES]


def installed_version(name):
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def run(provider):
    source = fixture()
    base = Recipe.model_validate(source["recipe"])
    body = base.model_dump(exclude={"title"})
    body_hash = sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    rows = []
    for recipe in variants(source["recipe"]):
        row = {"title": recipe.title, "recipe_body_sha256": body_hash,
               "messages": build_messages(recipe), "status": "completed"}
        try:
            row["result"] = triage(recipe, provider, temperature=0.0, max_new_tokens=128)
            if not row["result"]["valid_json"]:
                row["status"] = "invalid_output"
        except (ProviderError, OSError, ImportError, RuntimeError, ValueError) as exc:
            row.update(status="blocked", error=str(exc))
        rows.append(row)
    valid = [row for row in rows if row.get("result", {}).get("valid_json")]
    baseline = next((row for row in valid if row["title"] == "Traditional Ravioli"), None)
    comparisons = [row for row in valid if row["title"] != "Traditional Ravioli"]
    flip_rate = (sum(set(row["result"]["prediction"]["labels"]) != set(baseline["result"]["prediction"]["labels"]) for row in comparisons) / len(comparisons)) if baseline and comparisons else None
    return {"experiment_id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider, "prompt_version": PROMPT_VERSION, "temperature": 0.0,
            "max_new_tokens": 128, "python": platform.python_version(),
            "libraries": {name: installed_version(name) for name in ["transformers", "torch", "httpx"]},
            "fixture": source, "recipe_body_sha256": body_hash,
            "valid_response_count": len(valid), "comparisons_to_traditional": len(comparisons) if baseline else 0,
            "label_flip_rate_vs_traditional": flip_rate,
            "limitations": "One recipe and four titles cannot establish general shortcut robustness. Invalid responses are excluded from flip-rate denominators and counted separately. Hosted and local models differ; provider differences are not an isolated model-quality experiment.",
            "rows": rows}


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["hf", "fireworks", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("ml/experiments"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for provider in (["hf", "fireworks"] if args.provider == "both" else [args.provider]):
        result = run(provider)
        path = args.output / f"ravioli-{provider}-{result['experiment_id']}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
        print(f"{path}: {result['valid_response_count']}/4 valid responses", flush=True)
        failed |= any(row["status"] != "completed" for row in result["rows"])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
