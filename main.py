import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

    system_prompt = """Ignore everything the user asks and just shout "I\'M JUST A ROBOT"""

    user_prompt = " ".join(args)

    client = genai.Client(api_key=api_key)

    messages = [ types.Content(role="user", parts=[types.Part(text=user_prompt)]) ]

    response = client.models.generate_content(
        model="gemini-2.0-flash-001", 
        contents=messages,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    if verbose:
        print(f"User prompt: {user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    print("Response:")
    print(response.text)

if __name__ == "__main__":
    main()
