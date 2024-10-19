import os
import json

from openai import OpenAI
from dotenv import load_dotenv

import utils

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gpt_model = os.getenv("GPT_MODEL")


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


if __name__ == "__main__":
    directory_path = os.getenv('TRANSCRIPTS_PATH')
    files = os.listdir(directory_path)

    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(directory_path, file)

            process_transcript(utils.open_file(file_path))
