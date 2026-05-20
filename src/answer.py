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

    def find_function(self, prompt):
        generated_prompt = ""
        input_ids = self.llm.encode(prompt).tolist()[0]
        for _ in range(20):
            logit_data = self.llm.get_logits_from_input_ids(input_ids)
            logits_tensor = torch.tensor(logit_data)

            if logits_tensor.ndim == 1:
                next_token_id = torch.argmax(logits_tensor, dim=-1).item()
            else:
                next_token_id = torch.argmax(logits_tensor[-1, :],
                                             dim=-1).item()

            input_ids.append(next_token_id)
            decoded = self.llm.decode([next_token_id])
            generated_prompt += decoded
        return generated_prompt

    def generate_answer(self):
        function_description = self.get_function_description()
        result = []
        for line in self.prompt:
            prompt_all = (
                f"Available functions:\n{function_description}\n\n"
                "Select the first correct function with their parameters for:"
                f" {line.prompt}\n"
                f"Answer:"
            )
            function = self.find_function(prompt_all)
            result.append(
                {
                    "prompt": line.prompt,
                    "function": function,
                }
            )
        return result
