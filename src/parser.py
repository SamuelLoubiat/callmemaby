import json
from argparse import ArgumentParser
from typing import List

from src.data import Function, Prompt


class Parser:
    def __init__(self) -> None:
        default_definition = 'data/input/functions_definition.json'
        default_input = 'data/input/function_calling_tests.json'
        default_output = 'data/output/function_calling_results.json'
        self.parser = ArgumentParser(prog='callmemaby')
        self.parser.add_argument('--functions_definition',
                                 default=default_definition)
        self.parser.add_argument('--input', default=default_input)
        self.parser.add_argument('--output', default=default_output)
        self.args = self.parser.parse_args()

    def get_output(self) -> str:
        return self.args.output

    def get_functions(self) -> List[Function]:
        func = Function
        function = None
        with open(self.args.functions_definition, 'r') as f:
            function = [func(**item) for item in json.load(f)]
        return function

    def get_prompt(self) -> List[Prompt]:
        prompt_object = Prompt
        prompt = None
        with open(self.args.input, 'r') as f:
            prompt = [prompt_object(**item) for item in json.load(f)]
        return prompt
