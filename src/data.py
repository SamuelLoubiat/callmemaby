from typing import Dict

from pydantic import BaseModel, ConfigDict


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class Type(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, Type]
    returns: Type
