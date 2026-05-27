import json
import os

from src.answer import Answer
from src.parser import Parser


def get_arguments() -> tuple:
    parser = Parser()
    try:
        function_list = parser.get_functions()
        if len(function_list) == 0:
            raise Exception("Function not found")
    except Exception as e:
        print(e)
        exit(1)

    try:
        prompt_list = parser.get_prompt()
        if len(prompt_list) == 0:
            raise Exception("Prompt not found")
    except Exception as e:
        print(e)
        exit(1)

    return function_list, prompt_list, parser.get_output()


if __name__ == "__main__":
    try:
        function, prompt, output_path = get_arguments()
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as output:
                answer = Answer(function, prompt, output_path)
                output.write(json.dumps(answer.generate_answer(), indent=2))
        except Exception as e:
            print(e)
    except KeyboardInterrupt:
        print("User interrupt")
