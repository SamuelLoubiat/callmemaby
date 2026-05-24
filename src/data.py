from typing import Dict

from pydantic import BaseModel, ConfigDict


class Prompt(BaseModel):
    """Data validation schema for an input prompt configuration.

    Attributes:
        prompt (str): The raw evaluation query string containing execution
        targets.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str


class Type(BaseModel):
    """Data validation schema tracking data type definitions.

    Used to dynamically describe properties inside parameter signatures
    or execution return structures.

    Attributes:
        type (str): The datatype descriptor tag name (e.g., 'string',
        'integer', 'float').
    """
    model_config = ConfigDict(extra="forbid")
    type: str


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
    name: str
    description: str
    parameters: Dict[str, Type]
    returns: Type
