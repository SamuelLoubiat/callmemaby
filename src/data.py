from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo


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
    def check_keys(cls, v: Dict[str, str], info: ValidationInfo) \
            -> Dict[str, str]:
        """Validates that no keys in the parameters dictionary are empty or
        consist only of whitespace.

                    Args:
                        v (Dict[str, str]): The parameters dictionary to
                        validate.
                        info (ValidationInfo): ValidationInfo object.

                    Raises:
                        ValueError: If a key is found to be empty or contains
                        only whitespace.

                    Returns:
                        Dict[str, str]: The validated parameters dictionary.
                    """
        func_name = info.data.get('name', 'Unknown')
        print(v.keys())
        for key in v.keys():
            if not key.strip():
                raise ValueError(
                    f"Empty key detected in parameters for function"
                    f" {func_name}.")
        return v
