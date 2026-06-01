from src.llm import invoke_llm

HYDE_PROMPT = """
You are generating a hypothetical reference document for retrieval.

Question:
{question}

Write a concise but information-dense answer that likely contains
terminology and facts relevant to the question.
"""


def generate_hypothetical_document(question: str) -> str:
    prompt = HYDE_PROMPT.format(question=question)

    response = invoke_llm(prompt)

    if isinstance(response, str):
        return response

    # adjust depending on your LLM response format
    return response.content