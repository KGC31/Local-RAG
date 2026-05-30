list_command = {
    "/help": "Show commands",
    "/ingest": "Ingest a document",
    "/bye": "Exit the app",
}


def help_command():
    print("\nAvailable commands:")

    for cmd, desc in list_command.items():
        print(f"|   {cmd:<12} {desc}")


def ingest_command():
    print("\nIngesting document...")


def bye_command():
    print("\nGoodbye!")
    return False


COMMAND_HANDLERS = {
    "/help": help_command,
    "/ingest": ingest_command,
    "/bye": bye_command,
}


def handle_command(command: str) -> bool:
    handler = COMMAND_HANDLERS.get(command)

    if not handler:
        print(
            f"\nUnknown command: {command}"
        )
        return True

    result = handler()

    if result is False:
        return False

    return True