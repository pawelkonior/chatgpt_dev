import json
import os
from datetime import datetime, timedelta

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
