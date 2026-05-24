import json
from argparse import ArgumentParser
from typing import List, Any

from src.data import Function, Prompt


class Parser:
    """Handles command-line argument parsing and input data loading for the
    application.

    This class sets up defaults for function definitions, user inputs, and
    output paths,
    parses command-line execution arguments, and provides structured helper
    methods
    to deserialize JSON arrays directly into domain-specific objects.

    Attributes:
        parser (ArgumentParser): The underlying argument parser instance.
        args (Namespace): Parsed command-line arguments containing
        configuration paths.
    """

    def __init__(self) -> None:
        """Initializes the Parser, defining argument switches and parsing
        execution inputs."""
        default_definition = 'data/input/functions_definition.json'
        default_input = 'data/input/function_calling_tests.json'
        default_output = 'data/output/function_calling_results.json'
        self.parser = ArgumentParser(prog='callmemaby')
        self.parser.add_argument('--functions_definition',
                                 default=default_definition)
        self.parser.add_argument('--input', default=default_input)
        self.parser.add_argument('--output', default=default_output)
        self.args = self.parser.parse_args()

    def get_output(self) -> Any:
        """Retrieves the target configuration destination file path.

        Returns:
            Any: The string file path, or custom path stream matching target
            destination maps.
        """
        return self.args.output

    def get_functions(self) -> List[Function]:
        """Loads and unpacks function profile schemas from the configured JSON
        file definition.

        Returns:
            List[Function]: A collection of structural tool entities ready
                for downstream classification mapping.
        """
        with open(self.args.functions_definition, 'r') as f:
            function = [Function(**item) for item in json.load(f)]
        return function

    def get_prompt(self) -> List[Prompt]:
        """Loads and deserializes evaluation test datasets into distinct
        execution configurations.

        Returns:
            List[Prompt]: A list of mapped prompt elements defining evaluation
            test parameters.
        """
        with open(self.args.input, 'r') as f:
            prompt = [Prompt(**item) for item in json.load(f)]
        return prompt
