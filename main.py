import sys
from typing import List

from openai import OpenAI, OpenAIError
from pyaction import io

from pyaction.auth import Auth
from pyaction.issues import IssueForm


def text_operation(
    long_text,
    model="gpt-3.5-turbo",
    prompt="Summarized the text into two sentences",
):
    """
    Doing AI operations on the `long_text` by calling the MindsDB Serve API.

    :param long_text: Input text.
    :param prompt: Prompt to use for operation.
    :return: Summary of the text or an error message.
    """

    try:
        client_mindsdb_serve = OpenAI(
            api_key=io.read("mdb_token"),
            base_url="https://llm.mdb.ai",
        )
        chat_completion_gpt = client_mindsdb_serve.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": long_text},
            ],
            model=model,
        )
        return chat_completion_gpt.choices[0].message.content
    except OpenAIError as e:
        print(f"An error occurred with the MindsDB Serve API: {e}")
        return "Error in text operation."


def main(args: List[str]) -> None:
    """main function

    Args:
        args: STDIN arguments
    """

    auth = Auth(token=io.read("github_token"))

    repo = auth.github.get_repo(io.read("repository"))
    user_input = IssueForm(repo=repo, number=int(io.read("issue_number"))).render()

    text = user_input["Text"]
    prompt = user_input["Prompt"]

    output = text_operation(long_text=text, prompt=prompt)

    io.write({"answer": output})


if __name__ == "__main__":
    main(sys.argv[1:])
