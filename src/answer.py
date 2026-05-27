import json
import re
from typing import Dict, Any, List

from llm_sdk.llm_sdk import Small_LLM_Model
from src import Function, Prompt


class Answer:
    """Handles function selection and parameter extraction using a localized
    language model.

    This class processes user prompts by matching them against available
    predefined functions and dynamically extracting the corresponding
    parameters
    using rule-based parsing and LLM token constraints.

    Attributes:
        function (List[Function]): A list of available functional tools to
        match against.
        prompt (List[Prompt]): A list of input query data structure blocks.
        output (str): Destination path or configuration string for output
        tracking.
        llm (Small_LLM_Model): An instance of the underlying small language
        model.
        vocab (Dict[str, Any]): The vocabulary mapping tokens to IDs loaded
        from the model.
        id_to_token (Dict[int, str]): Reverse lookup dictionary mapping token
        IDs to string values.
        path_vocab (str): System file path to the vocabulary configuration
        json.
    """

    def __init__(self, function: List[Function], prompt: List[Prompt],
                 output: str) -> None:
        """Initializes the Answer instance with functions, prompts, and output
        references."""
        self.function = function
        self.prompt = prompt
        self.output = output
        self.llm = Small_LLM_Model()
        self.vocab: Dict[str, Any] = {}

    def get_vocab(self) -> None:
        """Loads the vocabulary map from the local language model.

        Populates the `vocab` and `id_to_token` attributes by loading the JSON
        schema provided by the underlying SDK path configuration.
        """
        self.path_vocab = self.llm.get_path_to_vocab_file()
        with open(self.path_vocab, 'r') as fd:
            self.vocab = json.load(fd)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def get_function_description(self) -> str:
        """Generates a structured string breakdown of all available tools.

        Returns:
            str: A newline-separated markdown string tracking function names
                and descriptions.
        """
        return "\n".join(
            f"- {d.name}: {d.description}" for d in
            self.function
        )

    def _fix_regex(self, value: str) -> str:
        """Balances trailing brackets and parentheses inside an isolated regex
        string.

        Args:
            value (str): The raw regex sequence parsed from the model.

        Returns:
            str: The sanitized regex pattern ensuring basic closing character
            parity.
        """
        if value.count('[') > value.count(']'):
            value += ']'
        if value.count('(') > value.count(')'):
            value += ')'
        return value

    def _extract_parameters(self, prompt: str, param_name: str,
                            context: str) -> str:
        """Uses a few-shot generative sequence to isolate target parameters.

        Args:
            prompt (str): The baseline query requested by the user.
            param_name (str): Key value descriptor (e.g., 'regex' or '
            replacement').
            context (str): System state history tracks containing previously
            evaluated flags.

        Returns:
            str: A stripped literal text output indicating the identified
            parameters.
        """
        end_tokens = set()
        for token_str, token_id in self.vocab.items():
            if 'Ċ' in token_str or '\n' in token_str:
                end_tokens.add(token_id)
        full_prompt_arg = (
            "Instruction: You are a strict few-shot string transformer."
            " Analyze the examples to understand "
            "the pattern, then complete the Last Task. Output ONLY the raw"
            " value for the requested parameter. "
            "Do not add quotes, do not explain, do not alter the examples.\n\n"

            "[EXAMPLE 1]\n"
            "Task: Replace all numbers in \"Hello 34\" with NUMBERS\n"
            "source_string: Hello 34 \n"
            "regex: \\d+ \n"
            "replacement: NUMBERS \n\n"

            "[EXAMPLE 2]\n"
            "Task: Replace all vowels in 'hello' with asterisks\n"
            "source_string: hello \n"
            "regex: [aeiouAEIOU] \n"
            "replacement: * \n\n"

            "[EXAMPLE 3] \n"
            "Task: Replace 'cat' with 'dog' in 'the cat sat'\n"
            "source_string: the cat sat \n"
            "regex: \\bcat\\b \n"
            "replacement: dog \n\n"

            "[LAST TASK]\n"
            f"Task: {prompt}\n"
            f"{context.strip()}\n"
            f"{param_name}: "
        )
        arg_generated = []
        for _ in range(30):
            test = self.llm.encode(full_prompt_arg).tolist()[0]
            logits = self.llm.get_logits_from_input_ids(test)
            max_logits: Any = logits.index(max(logits))
            if max_logits in end_tokens:
                break
            arg_generated.append(max_logits)
            full_prompt_arg += self.llm.decode(max_logits)
        result = self.llm.decode(arg_generated).strip()
        if param_name == "regex":
            result = self._fix_regex(result)
        return result

    def _find_parameters(self, prompt: str, chosen_function: Function) -> dict:
        """Parses arguments out of raw prompts matching targeted function
        schemas.

        Args:
            prompt (str): The raw instruction snippet containing entity tokens.
            chosen_function (Function): Meta-object validation profile mapping
            parameters.

        Returns:
            dict: Key-value bindings maps for positional tracking inside
            applications.
        """
        if chosen_function.name == "fn_substitute_string_with_regex":
            strings = [m[0] or m[1] for m in
                       re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)]
            params: Dict[str, Any] = {
                "source_string": max(strings, key=len) if strings else "",
            }
            context = f"source_string: {params['source_string']}\n"
            for param_name in ["regex", "replacement"]:
                value = self._extract_parameters(prompt, param_name, context)
                params[param_name] = value
                context += f"{param_name}: {value}\n"
            return params
        strings = [m[0] or m[1] for m in re.findall(
            r"'([^']*)'|\"([^\"]*)\"", prompt)]
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", prompt)
        params = {}
        num_idx = 0
        str_idx = 0
        for p_name, p_info in chosen_function.parameters.items():
            p_type = getattr(p_info, "type", "string")
            if p_type == "number" or p_type == "integer" or p_type == "float":
                if num_idx < len(numbers):
                    if p_type == "integer":
                        params[p_name] = int(numbers[num_idx])
                    else:
                        params[p_name] = float(numbers[num_idx])
                    num_idx += 1
                else:
                    params[p_name] = getattr(p_info, "default", 0.0)
            else:
                if str_idx < len(strings):
                    params[p_name] = strings[str_idx]
                    str_idx += 1
                else:
                    params[p_name] = getattr(p_info, "default", "")
        for p_name, p_val in params.items():
            if p_val == "" and len(prompt.split()) > 1:
                params[p_name] = prompt.strip().split()[-1].strip("?!.")
        return params

    def _find_function(self, prompt: str, functions: Dict[str, Function]) \
            -> str:
        """Finds valid matching tool tokens using constrained prefix logit
        masking.

        Args:
            prompt (str): Full target formatted contextual string schema.
            functions (Dict[str, Function]): Valid execution paths currently
            supported.

        Returns:
            str: The specific validated name key for an entity object tool.
        """
        input_ids = self.llm.encode(prompt).tolist()[0]
        generated_prompt = ""

        while True:
            if generated_prompt in functions.keys():
                break
            allowed_next_tokens = []
            for function in functions.keys():
                if function.startswith(generated_prompt):
                    remaining_text = function[len(generated_prompt):]
                    if remaining_text:
                        tokens = self.llm.encode(remaining_text).tolist()[0]
                        if tokens:
                            allowed_next_tokens.append(tokens[0])

            if not allowed_next_tokens:
                break

            logit_data = self.llm.get_logits_from_input_ids(input_ids)
            next_token_id = max(allowed_next_tokens,
                                key=lambda token_id: logit_data[token_id])

            input_ids.append(next_token_id)
            decoded = self.llm.decode(next_token_id)
            generated_prompt += decoded

        return generated_prompt

    def generate_answer(self) -> List:
        """Executes full pipeline resolution over the tracked array lists of
        prompts.

        Returns:
            List[Dict[str, Any]]: Array mapping prompts to their verified
                function structures and parameter variables.
        """
        function_description = self.get_function_description()
        result = []
        functions = {d.name: d for d in self.function}
        self.get_vocab()
        for line in self.prompt:
            prompt_all = (
                f"Available functions:\n{function_description}\n\n"
                "Select the first correct function with their parameters for:"
                f" {line.prompt}\n"
                f"Answer:"
            )
            function = self._find_function(prompt_all, functions)
            params = self._find_parameters(line.prompt, functions[function])
            result.append(
                {
                    "prompt": line.prompt,
                    "name": function,
                    "parameters": params,
                }
            )
        return result
