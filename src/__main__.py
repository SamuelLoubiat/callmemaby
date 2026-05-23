import json
import os

from src.answer import Answer
from src.parser import Parser

if __name__ == "__main__":
    parser = Parser()
    try:
        function = parser.get_functions()
    except Exception as e:
        print(e)
        exit(1)
    try:
        prompt = parser.get_prompt()
    except Exception as e:
        print(e)
        exit(1)
    output_path = parser.get_output()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = open(output_path, "w")
    answer = Answer(function, prompt, output_path)
    output.write(json.dumps(answer.generate_answer(), indent=2))
    output.close()
