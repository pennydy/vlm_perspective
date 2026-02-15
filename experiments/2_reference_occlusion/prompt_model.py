# import numpy as np
import pandas as pd
import logging
# import random
import base64
import re
from openai import OpenAI
from tqdm import tqdm
import argparse
import os
from google import genai
from google.genai import types
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger()
# client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# for gpt-5.1 and gpt-5.2
def get_response(prompt,model,reasoning_effort):
    if reasoning_effort == "none":
        reasoning_config = {
            "effort": reasoning_effort
        }
    elif reasoning_effort in ["low", "medium", "high"]:
        reasoning_config = {
            "effort": reasoning_config,
            "summary": "atuo"
        }
    else:
        raise TypeError("wrong reasoning type!")

    response = client.responses.create(
        model=model,
        input=prompt,
        # temperature=0, # not supported when reasoning is not none
        store=False,
        max_output_tokens = 256,
        text={
            "verbosity": "low"
            },
        reasoning=reasoning_config
        )
    if reasoning_effort != "none" and response.output[0].summary:
        thought_summary = response.output[0].summary[0].text
    else:
        thought_summary = "none"
        # print("no thoughts")
    generated_answer = response.output_text
    return generated_answer, thought_summary

# for gpt-4.1
def get_prediction(prompt, model, seed):
    prediction = client.chat.completions.create(
        model = model,
        messages = prompt,
        seed = seed,
        temperature = 0,
        max_tokens = 256,
        logprobs=True,
        top_logprobs=5 # ranging from 0 to 20, the number of most likely tokens to return at each token position
    )
    generated_answer = prediction.choices[0].message.content # text

    return generated_answer

# geting response
system_prompt_speaker = "You are the speaker in a reference game. Please use the shortest description possible." # Please use either one word or two words. # Please use the shortest description possible. # Please use only one word.
speaker_question = "Imagine you are talking to someone and want them to select the target object. The objects might be arranged differently for the other person, so please do not use degenerate spatial locations. Some objects are hidden behind a red curtain with a question mark, so you do not know what is behind, but the other person knows. The target image is highlighted by a dashed red box that only you can see."
system_prompt_listener = "Your are the listener in a reference game. Please only provide the column number and the row number."

# test on individual image
# # path to the image
# image_path = "exp2_001.jpeg"
# # getting the Base64 string
# base64_image = encode_image(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reference game")
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1") # or gpt-5.1
    parser.add_argument("--input", "-i", type=str, default="exp2_speaker.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="../../data/2_reference_occlusion")
    parser.add_argument("--task", "-t", type=str, default="speaker")
    parser.add_argument("--seed", "-s", type=int, default=1)
    parser.add_argument("--reasoning", "-r", type=str, default="low")
    args = parser.parse_args()

    prompts = pd.read_csv(args.input, header=0)
    task = args.task
    model = args.model
    reasoning_effort = args.reasoning

    for i, row in tqdm(prompts.iterrows()):
        image_file = row.image_file
        image_path = f"stimuli_{task}/"+image_file

        if task == "speaker":
            system_prompt = system_prompt_speaker
            question = speaker_question
        elif task == "listener":
            system_prompt = system_prompt_listener
            question = row.listener_question
        else:
            raise TypeError("wrong task type!")
        
        if model.startswith("gemini"):
            with open(image_path, 'rb') as f:
                image_byte = f.read()
            image = types.Part.from_bytes(
                data = image_byte,
                mime_type = "image/jpeg",
            )

            if model == "gemini-2.5-flash":
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=1,
                    thinking_config=types.ThinkingConfig(thinking_budget=2048,
                                                         include_thoughts=True)
                )
            elif model == "gemini-2.5-pro":
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=1, # recommended to be 1 but not deterministic
                    thinking_config=types.ThinkingConfig(thinking_budget=2048,
                                                         include_thoughts=True)
                )
            elif model == "gemini-3-flash-preview":
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level="low", # default is high
                                                         include_thoughts=True) 
                )
            elif model == "gemini-3-pro-preview":
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level="low",# default is high
                                                         include_thoughts=True) 
                )
            client = genai.Client()
            response = client.models.generate_content(
                model = model,
                config = config,
                contents =[image,question]
            )
            # get the thinking summary
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                elif part.thought:
                    thought_summary = part.text
                    break
                    # print("check the thought summary", thought_summary)
                    # print("check the text in the first part", response.candidates[0].content.parts[0].text)
                    # print("check the first elemnent response.parts", response.parts[0].text)
                else:
                    thought_summary = "none"
                    print("no thoughts")
            # print("final returned thought",thought_summary)
            generated_answer=response.text
            print("generated answer:",generated_answer)

        elif model.startswith("gpt"):
            # getting the Base64 string
            base64_image = encode_image(image_path)
            client = OpenAI()

            if model == "gpt-4.1":
                generated_answer = get_prediction(
                    prompt=[{"role" : "system", "content": system_prompt},
                             {"role": "user", "content": [
                                  {"type": "text", "text":question},
                                  {"type": "image_url", "image_url":{
                                       "url":f"data:image/jpeg;base64,{base64_image}"
                                         }}
                                         ]}],
                    model=model,
                    seed=args.seed
                )
                thought_summary = "none"
            elif model == "gpt-5.1" or model == "gpt-5.2":
                generated_answer, thought_summary = get_response(
                    prompt=[{"role": "system",
                             "content":[
                                 {"type" : "input_text", "text": system_prompt}
                             ]},
                             {"role": "user",
                              "content": [
                                  {"type": "input_text", "text": question},
                                  {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                              ]}],
                    model=model,
                    reasoning_effort = reasoning_effort
                )
            print("generated answer:", generated_answer)
            # print("thoughts:", thought_summary)

        elif model.startswith("qwen"):
            encoded_image = encode_image(image_path)
            model = AutoModelForImageTextToText.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
            )
            config = model.generation_config
            config.do_sample = False
            config.temperature = 0.0  # default temperature is 0.7
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            prompt=[{"role" : "system", 
                     "content": [{"type": "text", "text": system_prompt},]},
                     {"role": "user", 
                      "content": [
                            {"type": "text", "text":question},
                            {"type": "image", "image": encoded_image}]}]
            inputs = processor.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, 
                                           generation_config=config, # not sure this is working
                                           max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)

        prompts.loc[i, f"{task}_answer"] = generated_answer
        prompts.loc[i, f"{task}_thought"] = thought_summary

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prompts.to_csv(os.path.join(output_dir,f"{task}-{args.model}_{args.seed}_{args.reasoning}.csv"), index=False)