import os
import json
import sys

import openai
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import utils

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gpt_model = os.getenv("GPT_MODEL")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30, exp_base=2),
    retry=retry_if_exception_type((openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)),
)
def chat_completions_request(messages, model=gpt_model, json_mode=True, tools=None, tool_choice="auto"):
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": 0
    }

    if json_mode:
        api_params["response_format"] = {"type": "json_object"}

    if tools is not None:
        api_params["tools"] = tools
        api_params["tool_choice"] = tool_choice

    response = client.chat.completions.create(**api_params)
    return response.choices[0].message


def process_transcript(transcript):
    prompts_path = os.getenv("PROMPTS_PATH")

    if has_moderation_issues(transcript):
        sys.exit(1)
    else:
        print('Moderation passed')

    messages = [
        {"role": "system", "content": utils.open_file(os.path.join(prompts_path, "system_prompt.txt"))},
        {"role": "user", "content": utils.open_file(os.path.join(prompts_path, "user_prompt_01.txt")) + transcript},
    ]

    print("Information extracted... ⏰")

    first_response = chat_completions_request(messages)

    messages.append(first_response)

    info_object = json.loads(first_response.content.strip())

    messages.append({"role": "user", "content": utils.open_file(os.path.join(prompts_path, "user_prompt_02.txt"))})

    print("Information analyzed... 📈")
    second_response = chat_completions_request(messages)

    messages.append(second_response)

    json_to_append = json.loads(second_response.content.strip())

    info_object.update(json_to_append)

    utils.save_file(
        os.path.join("output", f"{info_object['candidate']}-{info_object['datetime']}.json"),
        json.dumps(info_object, indent=4)
    )

    print("Information saved! 💾")

    messages.append({"role": "user",
                     "content": "Please schedule a follow-up call"
                                " using the interview date extracted from the transcript."})

    third_response = chat_completions_request(messages, tools=utils.get_follow_up_function_desc())
    messages.append(third_response)

    tool_calls = third_response.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_message = tool_call.function

            if function_message.name == 'schedule_follow_up':
                args = json.loads(function_message.arguments)

                function_response = utils.schedule_follow_up(
                    interviewer=args.get("interviewer"),
                    candidate=args.get("candidate"),
                    interview_date=args.get("interview_date"),
                    sentiment=args.get("sentiment")
                )

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_message.name,
                    "content": function_response
                })

                fourth_response = chat_completions_request(messages, json_mode=False)
                messages.append(fourth_response)

    else:
        print("No function was called")

    utils.pretty_print_conversation(messages)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, exp_base=2, min=2, max=30),
    retry=retry_if_exception_type((openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError))
)
def has_moderation_issues(text):
    """
    Send a moderation request to OpenAI.

    Parameters
    ----------
        text (str): The text to moderate.
    """

    # Split the text into chunks
    chunks = utils.split_text_advanced(text)

    for chunk in chunks:
        # Send the API request and return the model's response.
        response = client.moderations.create(input=chunk)

        # Extract the value of flagged to see if the chunk violates OpenAI's content policy
        flagged = response.results[0].flagged
        if flagged:
            utils.format_moderation_response(response, chunk)
            return True

    return False


if __name__ == "__main__":
    directory_path = os.getenv('TRANSCRIPTS_PATH')
    files = os.listdir(directory_path)

    for file in files[:1]:
        if file.endswith(".txt"):
            file_path = os.path.join(directory_path, file)

            process_transcript(utils.open_file(file_path))
