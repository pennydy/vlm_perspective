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

logger = logging.getLogger()
# client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def get_prediction(prompt, model, seed):
    # prob = []
    prediction = client.chat.completions.create(
        model = model,
        messages = prompt,
        seed = seed,
        temperature = 0,
        max_tokens = 256,
        # max_completion_tokens = 256, # instead of max_tokens for gpt-5.1
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
    parser.add_argument("--input", "-i", type=str, default="exp2.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="../../data/2_reference_occlusion")
    parser.add_argument("--task", "-t", type=str, default="speaker")
    parser.add_argument("--seed", "-s", type=int, default=1)
    args = parser.parse_args()

    prompts = pd.read_csv(args.input, header=0)
    task = args.task
    model = args.model
    
    for i, row in tqdm(prompts.iterrows()):
        image_file = row.image_file
        image_path = f"stimuli_{task}/"+image_file

        if task == "speaker":
            system_prompt = system_prompt_speaker
            question = speaker_question
        if task == "listener":
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
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                # temperature=0 # recommended to be 1 so commented it out
            )
            client = genai.Client()
            response = client.models.generate_content(
                model = model,
                config = config,
                contents =[image,question]
            )
            generated_answer=response.text
            print(generated_answer)

        elif model.startswith("gpt"):
            # getting the Base64 string
            base64_image = encode_image(image_path)

            generated_answer = get_prediction(
                prompt=[{"role" : "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text":question},
                            {"type": "image_url", "image_url":{
                                "url":f"data:image/jpeg;base64,{base64_image}"
                            }}
                        ]}],
                seed=args.seed,
                model=model
            )
            print(generated_answer)
        prompts.loc[i, f"{task}_answer"] = generated_answer

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prompts.to_csv(os.path.join(output_dir,f"{task}-{args.model}_{args.seed}.csv"), index=False)