import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import system_prompt
from call_function import available_functions

def main():
    args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            args.append(arg)

    verbose = "--verbose" in sys.argv

    if not args: 
        print("AI Code Assistant")
        print('\nUsage: python main.py "your prompt here [--verbose]"')
        print('Example: python main.py "How do I build a calculator app?"')
        sys.exit(1)

    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")

    user_prompt = " ".join(args)

    client = genai.Client(api_key=api_key)

    messages = [ types.Content(role="user", parts=[types.Part(text=user_prompt)]) ]

    response = client.models.generate_content(
        model="gemini-2.0-flash-001", 
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[available_functions], system_instruction=system_prompt
        ),
    )

    if verbose:
        print(f"User prompt: {user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print("Response:")

    if not response.function_calls:
        print(response.text)

    for function_call_part in response.function_calls:
        print(f"Calling function: {function_call_part.name}({function_call_part.args})")


if __name__ == "__main__":
    main()
