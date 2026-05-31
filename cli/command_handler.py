import shlex

from sympy import arg

from src.ingest import ingest


list_command = {
    "/help": "Show commands",
    "/ingest": "Ingest a document",
    "/bye": "Exit the app",
}


def help_command():
    print("\nAvailable commands:")

    for cmd, desc in list_command.items():
        print(f"|   {cmd:<12} {desc}")


def ingest_command(*args): 
    if not args: 
        print( "\nUsage:\n" "/ingest file1.pdf file2.pdf " "--chunker semantic " "--chunk-size 1000 " "--chunk-overlap 150" ) 
        return 
    
    pdf_paths = [] 
    chunker = "recursive" 
    chunk_size = 1000 
    chunk_overlap = 150 
    i = 0 
    
    while i < len(args): 
        arg = args[i] 
 
        if arg == "--chunker": 
            chunker = args[i + 1] 
            i += 2 
        elif arg == "--chunk-size": 
            chunk_size = int(args[i + 1]) 
            i += 2 
        elif arg == "--chunk-overlap": 
            chunk_overlap = int(args[i + 1]) 
            i += 2 
        else: 
            pdf_paths.append(arg) 
            i += 1 
            
            if not pdf_paths: 
                print("\nNo PDF files provided.") 
                return 
            
        ingest( pdf_paths=pdf_paths, chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap, ) 
        print("\nIngestion complete.")


def bye_command():
    print("\nGoodbye!")
    return False


COMMAND_HANDLERS = {
    "/help": help_command,
    "/ingest": ingest_command,
    "/bye": bye_command,
}


def handle_command(user_input: str) -> bool: 
    parts = shlex.split(user_input) 
    
    if not parts: 
        return True 
    
    command = parts[0] 
    args = parts[1:] 
    handler = COMMAND_HANDLERS.get(command) 
    
    if not handler: 
        print(f"\nUnknown command: {command}") 
        return True 
    
    result = handler(*args) 
    if result is False: 
        return False 
    
    return True