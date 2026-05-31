from src.rag import answer
from src.llm import invoke_llm

def rag_answer_command(*args):
    if not args:
        print("\nUsage:\n/rag_answer [question] [--k 5] [--filter key=value ...]")
        return

    question = args[0]
    k = None
    filters = {}
    i = 1

    while i < len(args):
        arg = args[i]

        if arg == "--k":
            k = int(args[i + 1])
            i += 2
        elif arg == "--filter":
            filter_arg = args[i + 1]
            if "=" in filter_arg:
                key, value = filter_arg.split("=", 1)
                filters[key] = value
            else:
                print(f"Invalid filter format: {filter_arg}. Expected key=value.")
                return
            i += 2
        else:
            print(f"Unknown argument: {arg}")
            return

    answer_result = answer(
        question=question,
        k=k,
        filters=filters,
    )

    print("Answer:")
    print(answer_result.answer)

    if answer_result.citations:
        print("\nCitations:")
        for c in answer_result.citations:
            print(f"- {c.source_marker}: {c.filename} (page {c.page})")