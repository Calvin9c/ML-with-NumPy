from pathlib import Path
def is_file(pth: str | Path) -> bool:
    """
    Determines if the input string looks like a file path based on the presence of an extension.
    The file does not need to exist.
    
    Args:
        pth (str): The path string to evaluate.
        
    Returns:
        bool: True if the input string has a file extension, otherwise False.
    """
    
    if pth is None: return False
    if isinstance(pth, str):
        pth = Path(pth)

    return pth.suffix != ''  # Checks if there is a file extension

def is_dir(pth: str | Path) -> bool:
    """
    Determines if the input string looks like a directory path based on the absence of an extension.
    The directory does not need to exist.
    
    Args:
        pth (str): The path string to evaluate.
        
    Returns:
        bool: True if the input string does not have a file extension, otherwise False.
    """

    if pth is None: return False
    if isinstance(pth, str):
        pth = Path(pth)

    return pth.suffix == '' # Checks if there is no file extension

def mkdir(*directories: str | Path):
    """
    Creates directories based on input parameters.
    
    Parameters:
        *directories (str | Path): A variable number of inputs which can be strings or Path objects.
    """
    for directory in directories:

        # Check if it looks like a valid directory using the is_dir function
        if not is_dir(directory): continue # Skip invalid directory-like inputs
        if isinstance(directory, str):
            directory = Path(directory)

        # Check if the directory already exists, create it if it doesn't
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"Directory created: {directory}")
            except Exception as e:
                print(f"Failed to create directory {directory}: {e}")
        else:
            print(f"Directory already exists: {directory}")