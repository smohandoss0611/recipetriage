from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Label = Literal["weeknight-30min", "weekend-project", "needs-special-equipment", "meal-prep", "dessert", "have-most-of-this", "unclear"]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: Text
    ingredients: list[Text] = Field(min_length=1, max_length=60)
    instructions: list[Text] = Field(min_length=1, max_length=60)
    equipment: list[Text] = Field(max_length=30)
    time_minutes: int | None = Field(ge=1, le=10080)
    pantry_items: list[Text] | None = Field(default=None, max_length=100)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    labels: list[Label] = Field(min_length=1, max_length=7)
    explanation: Text

    @model_validator(mode="after")
    def labels_valid(self):
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("Duplicate labels are invalid")
        if "unclear" in self.labels and len(self.labels) != 1:
            raise ValueError("unclear cannot accompany other labels")
        return self


class Generation(BaseModel):
    raw_output: str
    model: str
    revision: str | None = None
    finish_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderError(RuntimeError):
    """Safe, user-visible configuration, capacity, or upstream error."""
