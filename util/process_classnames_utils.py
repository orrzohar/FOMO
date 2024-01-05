# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import json
import ast


def read_file(file_path:str) -> list:
    """
    Reads a file and appends each line to a list.

    Args:
    file_path (str): The path to the file to be read.

    Returns:
    list: A list containing each line from the file.
    """
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(line.strip())  # Append each line, removing any extra whitespace

    return data_list


def save_to_file(iterable:set, file_path: str) -> None:
    """
    Saves the content of an iterable to a specified file.

    Args:
    iterable (iterable): The iterable to be saved to the file.
    file_path (str): The path where the content will be saved.
    """
    with open(file_path, 'w') as file:
        for item in iterable:
            file.write(str(item) + '\n')  # Writing each item from the iterable to the file, adding a newline

def get_first_item_from_txt(file_path:str):
    """
    Extracts the first item of lists associated with keys in a JSON file.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict or None: A dictionary containing keys from the JSON file and their associated first list items.
                 Returns None if an error occurs during file processing.

    Note:
    If a list associated with a key is empty, its corresponding value in the returned dictionary will be None.
    """
    elements = []
    try:    

        with open(file_path, 'r') as file:
            data      = file.read()
            data_dict = ast.literal_eval(data)
     
        first_items = [ values.split(",")[0].lower().replace("-"," ").replace("_"," ") if values else None for key, values in data_dict.items()]

        return first_items

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return None


def get_first_item_from_json(file_path:str):
    """
    Extracts the first item of lists associated with keys in a JSON file.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict or None: A dictionary containing keys from the JSON file and their associated first list items.
                 Returns None if an error occurs during file processing.

    Note:
    If a list associated with a key is empty, its corresponding value in the returned dictionary will be None.
    """
    elements = []
    try:    
        with open(file_path, 'r') as file:
            data = json.load(file)
            first_items = [ values[0] if values else None for key, values in data.items()]
            return first_items

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return None
