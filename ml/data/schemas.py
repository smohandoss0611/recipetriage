"""Dataset contracts extend the same recipe fields used by inference."""
from typing import Annotated, Literal
import unicodedata
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from recipetriage_ml.inference.contracts import Recipe as InferenceRecipe, Label

LABELS = ["weeknight-30min", "weekend-project", "needs-special-equipment", "meal-prep", "dessert", "have-most-of-this", "unclear"]
POLICY = {
    "weeknight-30min": "Known total elapsed time <=30 minutes, including waiting; never infer speed from title adjectives.",
    "weekend-project": "A substantial cooking project taking >30 minutes; passive waiting alone does not qualify.",
    "needs-special-equipment": "Requires a specialized appliance, such as a blender, food processor, pasta machine, or waffle iron; ordinary kitchen tools do not qualify.",
    "meal-prep": "Explicitly suitable for preparing portions ahead; record recipe-grounded evidence.",
    "dessert": "Intended as dessert, not merely sweet.",
    "have-most-of-this": "Explicit pantry context covers >=80% of ingredient items; never assume the user's pantry. Dataset v1 uses exact normalized ingredient strings for matching.",
    "unclear": "Insufficient or contradictory evidence; exclusive of all other labels in v1.",
}
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,80}$")]


def clean(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


class Recipe(InferenceRecipe):
    id: Identifier
    source_type: Literal["published", "personal", "synthetic"]
    source_uri: Text
    source_notes: Text
    group_id: Identifier | None = None

    @field_validator("title", "source_uri", "source_notes", mode="before")
    @classmethod
    def clean_text(cls, value):
        return clean(value) if isinstance(value, str) else value

    @field_validator("ingredients", "instructions", "equipment", "pantry_items", mode="before")
    @classmethod
    def clean_lists(cls, value):
        return [clean(x) if isinstance(x, str) else x for x in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def valid_source(self):
        if self.source_type == "published":
            uri = urlsplit(self.source_uri)
            if uri.scheme not in ("http", "https") or not uri.hostname or uri.username or uri.password:
                raise ValueError("Published recipes require an HTTP(S) source URL without credentials")
        return self

    def inference_recipe(self):
        return InferenceRecipe.model_validate(self.model_dump(include=set(InferenceRecipe.model_fields)))


class SyntheticProvenance(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    source: Literal['synthetic-fireworks']
    generation_id: str
    candidate_id: str
    candidate_revision: int = Field(ge=1)
    model: Text
    prompt_version: Text
    prompt_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    human_review_id: str
    approved_at: Text
    approved_content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    recipe: Recipe
    labels: list[Label] = Field(min_length=1, max_length=7)
    rationale: Text
    annotation_notes: str = Field(default="", max_length=4000)
    reviewed: bool = False
    reviewed_by: Text | None = None
    provenance: SyntheticProvenance | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("rationale", "annotation_notes", "reviewed_by", mode="before")
    @classmethod
    def clean_text(cls, value):
        return clean(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def consistent_labels(self):
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("Repeated labels are invalid")
        if "unclear" in self.labels and len(self.labels) != 1:
            raise ValueError("unclear must be used alone")
        if "weeknight-30min" in self.labels and (self.recipe.time_minutes is None or self.recipe.time_minutes > 30):
            raise ValueError("weeknight-30min requires known total time <=30 minutes")
        if "weekend-project" in self.labels and (self.recipe.time_minutes is None or self.recipe.time_minutes <= 30):
            raise ValueError("weekend-project requires known total time >30 minutes")
        if "needs-special-equipment" in self.labels and not self.recipe.equipment:
            raise ValueError("Specify the required equipment")
        if "have-most-of-this" in self.labels:
            needed = {clean(x).casefold() for x in self.recipe.ingredients}
            available = {clean(x).casefold() for x in self.recipe.pantry_items or []}
            if len(needed & available) / len(needed) < 0.8:
                raise ValueError("have-most-of-this requires pantry_items matching >=80% of ingredient items")
        if self.reviewed and not self.reviewed_by:
            raise ValueError("reviewed_by is required when reviewed=true")
        if not self.reviewed and self.reviewed_by is not None:
            raise ValueError("Unreviewed examples cannot claim a reviewer")
        self.labels = sorted(self.labels)
        if self.recipe.source_type == 'synthetic':
            if not self.reviewed or self.provenance is None:
                raise ValueError('Synthetic examples require explicit human approval and provenance before export')
            from .pipeline import digest
            content = {'recipe': self.recipe.model_dump(), 'labels': self.labels,
                       'rationale': self.rationale, 'annotation_notes': self.annotation_notes}
            if digest(content) != self.provenance.approved_content_sha256:
                raise ValueError('Synthetic content changed after approval; return it to the review queue')
        elif self.provenance is not None:
            raise ValueError('Synthetic provenance must retain synthetic source_type')
        return self
