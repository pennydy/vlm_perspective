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
# system_prompt = "You are the speaker in a reference game."
system_prompt_speaker = "Your job is to decide what utterance to use. Imagine that you have $100. You should divide your money between the possible utterances -- the amount of money you bet on each option should correspond to how confident you are that it will lead the listener to the correct choice. Bets must sum to 100, which means you have to place bets. You do not need to provide any reasoning." # Please provide it in the format of 'word1': money; 'word2':money.
# question_speaker = "Imagine you are talking to someone and you want to refer to the middle object. Which word would you use, 'blue' or 'circle'?"
# system_prompt_free_speaker = "Imagine you are talking to someone and want them to select the target object, but the objects might be arranged differently for the other person. Your job is to decide what utterance to use. So please avoid using absolute positions and the label. The target image is highlighted by a dashed red box that only you can see. Please use only one word." # Please use either one word or two words. # Please use the shortest description possible. # Please use only one word.
system_prompt_free_speaker = "You are the speaker in a reference game. Please use the shortest description possible."  # Please use either one word or two words. # Please use the shortest description possible. # Please use only one word.
system_prompt_listener = "Your job is to decide which object the speaker is talking about. Imagine that you have $100. You should divide your money between the possible objects -- the amount of money you bet on each option should correspond to how confident you are that it is correct. Bets must sum to 100. You do not need to provide any reasoning." # Please provide it in the format of 'word1': money; 'word2':money.
# question_listener = "Imagine someone is talking to you and uses the word 'square' to refer to one of the objects. Which object do you think they are talking about?"
question_prior = "Imagine someone is talking to you and uses a word you don't know to refer to one of the objects. Which object do you think they are talking about?"

# system_prompt = "Imagine you are talking to someone and want them to select the target object, but the objects might be arranged differently for the other person. So please avoid using absolute positions, and please use either a single word a two-word phrase."
# question = "You want to refer to the middle object."

# test on individual image
# # path to the image
# image_path = "stimuli_001.jpeg"
# # getting the Base64 string
# base64_image = encode_image(image_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="reference game")
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1") # or gpt-5.1
    parser.add_argument("--input", "-i", type=str, default="exp1_cond4.csv")
    parser.add_argument("--output_dir", "-o", type=str, default="../../data/")
    parser.add_argument("--condition", "-c", type=str, default="cond1")
    parser.add_argument("--task", "-t", type=str, default="speaker")
    parser.add_argument("--seed", "-s", type=int, default=1)
    args = parser.parse_args()

    prompts = pd.read_csv(args.input, header=0)
    task = args.task
    condition = args.condition
    
    for i, row in tqdm(prompts.iterrows()):
        image_file = row.image_file
        image_path = f"stimuli_{condition}/"+image_file
        word_1 = row.adj1
        word_2 = row.adj2
        question_speaker = row.question_speaker
        question_free_speaker = row.question_free_speaker
        question_listener_1 = row.question_listener_1
        question_listener_2 = row.question_listener_2

        if task == "speaker":
            system_prompt = system_prompt_speaker
            question = question_speaker
        elif task == "free_speaker":
            system_prompt = system_prompt_free_speaker + " Please use the shortest description possible."
            question = question_free_speaker
        elif task == "prior":
            system_prompt = system_prompt_listener
            question = question_prior
        elif task == "listener1":
            system_prompt = system_prompt_listener
            question = question_listener_1
        elif task == "listener2":
            system_prompt = system_prompt_listener
            question = question_listener_2
        else:
            raise TypeError("wrong task type!")

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
            model=args.model
        )
        print(generated_answer)
        prompts.loc[i, f"{task}_answer"] = generated_answer

    output_dir = args.output_dir
    prompts.to_csv(os.path.join(output_dir,f"{task}-{condition}-{args.model}_{args.seed}.csv"), index=False)