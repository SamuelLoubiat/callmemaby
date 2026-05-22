from typing import List

import torch

from llm_sdk.llm_sdk import Small_LLM_Model


class Answer:
    def __init__(self, function, prompt, output) -> None:
        self.function = function
        self.prompt = prompt
        self.output = output
        self.llm = Small_LLM_Model()

    def get_function_description(self) -> str:
        return "\n".join(
            f"- {d.name}: {d.description}" for d in
            self.function
        )

    def find_function(self, prompt, functions: List[str]):
        input_ids = self.llm.encode(prompt).tolist()[0]
        generated_prompt = ""

        while True:
            if generated_prompt in functions:
                break
            allowed_next_tokens = []
            for function in functions:
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
        functions = [d.name for d in self.function]
        for line in self.prompt:
            prompt_all = (
                f"Available functions:\n{function_description}\n\n"
                "Select the first correct function with their parameters for:"
                f" {line.prompt}\n"
                f"Answer:"
            )
            function = self.find_function(prompt_all, functions)
            result.append(
                {
                    "prompt": line.prompt,
                    "function": function,
                }
            )
        return result
