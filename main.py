from openai import OpenAI, OpenAIError
from pyaction import PyAction
from pyaction.auth import Auth
from pyaction.issues import IssueForm

from pyaction import io


workflow = PyAction()


def text_operation(
    long_text,
    model="gpt-3.5-turbo",
    prompt="Summarize the text into one sentence",
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
        return f"Error in text operation. {e}"


@workflow.action()
def main(github_token: str, repository: str, issue_number: int) -> None:
    auth = Auth(token=github_token)

    repo = auth.github.get_repo(repository)
    user_input = IssueForm(repo=repo, number=issue_number).render()

    text = user_input["Text"]
    prompt = user_input["Prompt"]

    output = text_operation(long_text=text, prompt=prompt)

    workflow.write({"answer": output})
