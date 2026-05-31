from pathlib import Path
import os
import warnings

from cli.command_handler import handle_command


HEADER_FILE = "header.txt"


def load_header(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        return "RAG Chunking CLI"

    return path.read_text(encoding="utf-8")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    clear_screen()

    header = load_header(HEADER_FILE)

    print(header)
    print("-" * 50)
    print("Type /help for commands")
    print("-" * 50)


def main():
    print_header()
    warnings.filterwarnings( "ignore", message="Payload indexes have no effect in the local Qdrant.*", )

    try:
        while True:
            user_input = input("\n> ").strip()

            if not user_input:
                print("> Please enter a command or query.")
                continue
            
            if user_input.startswith("/"):
                should_continue = handle_command(user_input)
                if not should_continue:
                    break
            else:
                print(f"\nYou entered:\n{user_input}")
                

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        print("Exiting...")


if __name__ == "__main__":
    main()