from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Prompt(BaseModel):
    """Data validation schema for an input prompt configuration.

    Attributes:
        prompt (str): The raw evaluation query string containing execution
        targets.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class Type(BaseModel):
    """Data validation schema tracking data type definitions.

    Used to dynamically describe properties inside parameter signatures
    or execution return structures.

    Attributes:
        type (str): The datatype descriptor tag name (e.g., 'string',
        'integer', 'float').
    """
    model_config = ConfigDict(extra="forbid")
    type: str = Field(min_length=1)


class Function(BaseModel):
    """Data validation schema defining executable metadata components.

    Represents structural function specifications used to dynamically bound
    and parse target attributes matching tool execution manifests.

    Attributes:
        name (str): The target unique system function registry identifier key.
        description (str): Explanatory documentation text detailing utility
        behaviors.
        parameters (Dict[str, Type]): A dictionary mapping structural parameter
            names to their property data configurations.
        returns (Type): The specified evaluation return output data profile
        schema.
    """
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Type]
    returns: Type

    @field_validator('parameters')
    @classmethod
    def verifier_cles(cls, v: Dict[str, str]) -> Dict[str, str]:
        for cle in v.keys():
            if not cle.strip():
                raise ValueError(
                    f"Empty key detected in parameters for function {cls.name}.")
        return v
