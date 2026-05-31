import ollama


def get_llm(model: str = "llama3"):

    return {
        "model": model,
    }


def invoke_llm(prompt: str, model: str = "qwen2.5:7b", system: str | None = None,):
    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system,
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    response = ollama.chat(
        model=model,
        messages=messages,
    )

    return response["message"]["content"]
