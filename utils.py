import json
import os
import re
import textwrap
from datetime import datetime, timedelta

import tiktoken
from openai.types.chat import ChatCompletionMessage
from termcolor import colored


def open_file(file_path):
    """
    Opens a file and returns its content as a string.

    Args:
        file_path (str): The path to the file to be opened.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        IOError: If an error occurs while opening or reading the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def save_file(file_path, content):
    """
    Saves the provided content to a file at the specified file path.
    If the directory does not exist, it is created.

    Args:
        file_path (str): The path where the file will be saved.
        content (str): The content to write into the file.

    Raises:
        IOError: If an error occurs during the file writing process.
    """
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def get_scheduled_date(date_str):
    """
    Takes an ISO 8601 formatted date string, adds 7 days to it, and returns the new date as a string.

    Args:
        date_str (str): The original date and time in ISO 8601 format ("YYYY-MM-DDTHH:MM:SSZ").

    Returns:
        str: The new date and time, 7 days later, in the same ISO 8601 format.

    Raises:
        ValueError: If the input date string does not match the expected format.
    """
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    date_object = datetime.strptime(date_str, date_format)

    new_date = date_object + timedelta(days=7)

    return new_date.strftime(date_format)


def schedule_follow_up(interviewer, candidate, interview_date, sentiment=None):
    """
    Schedules a follow-up call by calculating a new date 7 days after the given interview date and returning the details
    in JSON format. Optionally includes the sentiment of the interview.

    Args:
        interviewer (str): The name of the interviewer.
        candidate (str): The name of the candidate.
        interview_date (str): The date and time of the interview in ISO 8601 format ("YYYY-MM-DDTHH:MM:SSZ").
        sentiment (str, optional): The sentiment of the interview. Can be one of "positive", "negative", or "neutral".
                                   Defaults to None.

    Returns:
        str: A JSON string containing the details of the scheduled follow-up, including:
            - interviewer (str): The name of the interviewer.
            - candidate (str): The name of the candidate.
            - schedule_date (str): The new follow-up date, 7 days after the interview, in ISO 8601 format.
            - sentiment (str, optional): The sentiment of the interview, if provided.

    Raises:
        ValueError: If the interview_date is not in the correct ISO 8601 format.
    """
    new_date_str = get_scheduled_date(interview_date)

    response = {
        "interviewer": interviewer,
        "candidate": candidate,
        "schedule_date": new_date_str,
    }

    if sentiment is not None:
        response["sentiment"] = sentiment

    return json.dumps(response)


def get_follow_up_function_desc():
    """
    Returns a description of the follow-up function, including the parameters needed for scheduling a follow-up call.

    The function description is structured as a list containing a dictionary that defines the function type, name, description,
    and parameters. The parameters describe details about the follow-up call, such as the interviewer's name, candidate's name,
    interview date, and sentiment of the interview.

    Returns:
        list: A list containing a dictionary that describes the follow-up function, including:
            - name (str): The name of the function ("schedule_follow_up").
            - description (str): A brief description of the function's purpose.
            - parameters (dict): An object containing the required and optional parameters for the follow-up function:
                - interviewer (str): The name of the interviewer.
                - candidate (str): The name of the candidate.
                - interview_date (str): The date and time of the interview in ISO 8601 format.
                - sentiment (str): The sentiment of the interview (can be "positive", "negative", or "neutral").
            - required (list): The parameters that must be provided ("interviewer", "candidate", "interview_date").
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "schedule_follow_up",
                "description": "Get the details of the scheduled follow-up call",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interviewer": {
                            "type": "string",
                            "description": "The name of the interviewer",
                        },
                        "candidate": {
                            "type": "string",
                            "description": "The name of the candidate",
                        },
                        "interview_date": {
                            "type": "string",
                            "format": "date-time",
                            "description": "The date and time of the interview in ISO 8601 format",
                        },
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"],
                            "description": "The sentiment of the interview",
                        }
                    },
                    "required": ["interviewer", "candidate", "interview_date"]
                }
            }
        }
    ]


def pretty_print_conversation(messages):
    """
    Prints a formatted conversation with colored roles for better readability.

    This function takes a list of message objects, processes them, and prints each message
    in a formatted and colored output depending on the role of the sender. Supported roles
    include 'system', 'user', 'assistant', and 'tool'. The colors are:

    - 'system' messages are printed in magenta.
    - 'user' messages are printed in green.
    - 'assistant' messages are printed in yellow. If the message includes a tool call, the
      tool call content is printed instead of the message content.
    - 'tool' messages are printed in blue, showing the tool's name and content.

    Args:
        messages (list): A list of message objects, where each message is either a dictionary
                         or a `ChatCompletionMessage` object. Each message must contain at least
                         a 'role' and 'content', with optional keys such as 'tool_calls' for
                         assistant messages and 'name' for tool messages.

    Raises:
        AttributeError: If a message is not in the expected format or missing required fields.
    """
    role_to_color = {
        "system": "magenta",
        "user": "green",
        "assistant": "yellow",
        "tool": "blue"
    }

    for raw_message in messages:
        message = raw_message

        if isinstance(raw_message, ChatCompletionMessage):
            message = raw_message.dict()

        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("tool_calls"):
            print(colored(f"assistant: {message['tool_calls']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("tool_calls"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "tool":
            print(colored(f"function: {message['name']}: {message['content']}\n", role_to_color[message["role"]]))


def split_text_simple(text, length_limit=2000):
    """
    Splits a given text into chunks each with maximum length as length_limit.
    This function splits text without considering paragraph structure.

    Parameters
    ----------
    text : str
        The text to split.
    length_limit : int, optional
        The maximum character length of each chunk (default is 2000).

    Returns
    -------
    list of str
        The list of split chunks.
    """
    return textwrap.wrap(text, length_limit)


def split_text_advanced(text, length_limit=2000):
    """
    Splits a given text into chunks each with maximum length as length_limit.
    This function splits text considering paragraph structure.

    Parameters
    ----------
    text : str
        The text to split.
    length_limit : int, optional
        The maximum character length of each chunk (default is 2000).

    Returns
    -------
    list of str
        The list of split chunks respecting paragraph boundaries.
    """
    # Split text into paragraphs using single or double line breaks
    paragraphs = re.split(r'\n\n|\n(?!\n)', text)
    lines = []

    for paragraph in paragraphs:
        # Split each paragraph into lines with maximum length as length_limit
        lines.extend(textwrap.wrap(paragraph, length_limit))

    return lines


def format_moderation_response(response, text):
    """
    Formats and prints the moderation response from OpenAI's API.

    This function takes the response from the moderation API and the text that was checked.
    It prints whether the text violates the content policy, the model used for the check,
    the categories that were evaluated, and the corresponding scores for each category.

    It does not return any value.

    Parameters
    ----------
    response : object
        The response object returned from OpenAI's moderation API.
                         This should contain the results of the content moderation check,
                         including categories and category scores.
    text : str
        The text that was submitted for moderation.
    """

    # Extracting categories and category scores from the response
    categories = response.results[0].categories
    category_scores = response.results[0].category_scores

    # Informing if the text violates the content policy
    print(f"Text violates content policy: \n{text}\n")

    # Printing the model used for moderation
    print(f"Moderation model: {response.model}\n")

    # Iterating over the categories to print them
    print("Categories:")
    for category, value in categories.__dict__.items():
        # Formatting category names and printing their values
        print(f"  {category.replace('_', ' ').capitalize()}: {value}")

    # Iterating over the category scores to print them
    print("\nCategory Scores:")
    for category, score in category_scores.__dict__.items():
        # Formatting category names and printing their scores
        print(f"  {category.replace('_', ' ').capitalize()}: {score:.2f}")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        message_dict = message
        if isinstance(message, ChatCompletionMessage):
            message_dict = message.model_dump()
        for key, value in message_dict.items():
            num_tokens += len(encoding.encode(get_safe_string(value)))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_safe_string(value):
    """
    Converts the input value to a string in a safe manner.

    Parameters
    ----------
    value : str
        The value to be converted to string.

    Returns
    ----------
    A string representation of the input value.
    """

    if value is None:  # If the value is None
        return ''  # Return an empty string
    elif isinstance(value, str):  # If the value is already a string
        return value  # Return the value as is
    elif isinstance(value, dict) or isinstance(value, list):  # If the value is a JSON object or array
        return json.dumps(value)  # Convert JSON object/array to string and return
    else:  # For all other data types
        return str(value)  # Convert the value to string and return


def print_token_info(messages, model, response, price_per_input_token, price_per_output_token):
    """
    Prints the token count and cost information for the given messages and response.

    Parameters
    ----------
    messages:
        The messages for which the token count is to be calculated.
    model:
        The model used to calculate the token count.
    response:
        The response received from the OpenAI API, which includes the usage data.
    price_per_input_token:
        The cost per input token.
    price_per_output_token:
        The cost per output token.
    """

    # Calculate the number of tokens from the messages
    tokens = num_tokens_from_messages(messages, model)
    # Calculate the cost for the input tokens
    cost = price_per_input_token * tokens
    # Print the token count and cost
    print(f"{tokens} prompt tokens counted by tiktoken. Cost: ${cost}")

    # Get the number of prompt tokens from the response
    prompt_tokens = response.usage.prompt_tokens
    # Calculate the cost for the prompt tokens
    cost = price_per_input_token * prompt_tokens
    # Print the token count and cost
    print(f'{prompt_tokens} prompt tokens counted by the OpenAI API. Cost: ${cost}')

    # Get the number of completion tokens from the response
    completion_tokens = response.usage.completion_tokens
    # Calculate the cost for the completion tokens
    cost = price_per_output_token * completion_tokens
    # Print the token count and cost
    print(f'{completion_tokens} completion tokens counted by the OpenAI API. Cost: ${cost}')
