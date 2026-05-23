import json
import re
from typing import Dict, Any

import torch

from llm_sdk.llm_sdk import Small_LLM_Model
from src import Function


class Answer:
    def __init__(self, function, prompt, output) -> None:
        self.function = function
        self.prompt = prompt
        self.output = output
        self.llm = Small_LLM_Model()
        self.vocab: Dict[str, Any] = {}

    def get_vocab(self) -> None:
        self.path_vocab = self.llm.get_path_to_vocab_file()
        with open(self.path_vocab, 'r') as fd:
            self.vocab = json.load(fd)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def get_function_description(self) -> str:
        return "\n".join(
            f"- {d.name}: {d.description}" for d in
            self.function
        )

    def _fix_regex(self, value: str) -> str:
        if value.count('[') > value.count(']'):
            value += ']'
        if value.count('(') > value.count(')'):
            value += ')'
        return value

    def _extract_parameters(self, prompt: str, param_name: str,
                          context: str) -> str:
        newline_id = self.llm.encode('\n').tolist()[0][0]
        double_newline_id = self.llm.encode('\n\n').tolist()[0][0]
        end_tokens = {newline_id, double_newline_id}
        for token_str, token_id in self.vocab.items():
            if 'Ċ' in token_str or '\n' in token_str:
                end_tokens.add(token_id)
        full_prompt_arg = (
            f"Examples:\n"
            f"Task: Replace all numbers in 'Hello 42' with NUMBERS\n"
            f"source_string: Hello 42\n"
            f"regex: '\\d+'\n"
            f"replacement: 'NUMBERS'\n\n"
            f"Task: Replace vowels in 'hello' with *\n"
            f"source_string: hello\n"
            f"regex: '[aeiouAEIOU]'\n"
            f"replacement: *\n\n"
            f"Task: Replace 'cat' with 'dog' in 'the cat sat'\n"
            f"source_string: the cat sat\n"
            f"regex: \\bcat\\b\n"
            f"replacement: 'dog'\n\n"
            f"Task: {prompt}\n"
            f"{context}"
            f"{param_name}: "
        )
        arg_generated = []
        for _ in range(30):
            test = self.llm.encode(full_prompt_arg).tolist()[0]
            logits = self.llm.get_logits_from_input_ids(test)
            max_logits = logits.index(max(logits))
            if max_logits in end_tokens:
                break
            arg_generated.append(max_logits)
            full_prompt_arg += self.llm.decode(max_logits)
        result = self.llm.decode(arg_generated).strip()
        if param_name == "regex":
            result = self._fix_regex(result)
        return result

    def _find_parameters(self, prompt, chosen_function):
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

    def _find_function(self, prompt, functions: Dict[str, Function]):
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
            logits_tensor = torch.tensor(logit_data)

            mask = torch.full(logits_tensor.shape, float('-inf'))
            mask[allowed_next_tokens] = 0.0

            constrained_logits = logits_tensor + mask
            next_token_id = torch.argmax(constrained_logits, dim=-1).item()

            input_ids.append(next_token_id)
            decoded = self.llm.decode([next_token_id])
            generated_prompt += decoded

        return generated_prompt

    def generate_answer(self):
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
