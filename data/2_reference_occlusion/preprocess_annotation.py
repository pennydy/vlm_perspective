from ast import pattern
import pandas as pd
import argparse
import re

def check_features(row):
    answer = row["speaker_answer"]
    return pd.Series({
        "contain_shape": any(word in answer for word in row["target_shape_list"]),
        "contain_color": row["target_color"] in answer,
        "contain_texture": any(word in answer for word in row["target_texture_list"]),
    })

def get_numbers(listener_answer):
    listener_answer_number = re.findall(r"\d+", str(listener_answer))
    if len(listener_answer_number) == 2:
        listener_answer=f"{listener_answer_number[0]},{listener_answer_number[1]}"
        return listener_answer
    else:
        print("the intput is longer than 2 numbers")
        return listener_answer
    
def extract_thought_answer(full_answer):
    full_answer = str(full_answer)
    thought_answer = re.search(r"(.*?)</think>", full_answer, re.DOTALL) 
    if thought_answer:
        thoughts = thought_answer.group(1).lstrip("\n")
        answer = full_answer.split("</think>", 1)[1].lstrip("\\n")
        return thoughts, answer
    if "<think>" in full_answer:
        thoughts = full_answer.split("<think>", 1)[1].lstrip("\\n")
        print("Answer not found in the generated text.")
        return thoughts, ""
    print("full_answer:", full_answer)
    return full_answer.strip(), ""


def remove_symbols(text):
    if not pd.isna(text):
        for pattern in ["\\n", "['", "']", "<think>", "\""]:
            text = text.replace(pattern, "")
    return text.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess and annotate results")
    parser.add_argument("--input", "-i", type=str, default="speaker-gpt-5.2_1_none.csv")
    parser.add_argument("--task", "-t", type=str, default="speaker")

    args = parser.parse_args()
    task = args.task

    file_name = re.sub(r"\.csv$", "", args.input)

    results = pd.read_csv(args.input, header=0)

    if task == "speaker":
        if "qwen" in file_name.lower() and "thinking" in file_name.lower():
            results["speaker_answer_original"] = results["speaker_answer"]
            results["speaker_answer"] = results["speaker_answer_original"].apply(lambda x: extract_thought_answer(x)[1])
            results["speaker_thought"] = results["speaker_answer_original"].apply(lambda x: extract_thought_answer(x)[0])

            results["speaker_answer"] = results["speaker_answer"].apply(remove_symbols)
            results["speaker_thought"] = results["speaker_thought"].apply(remove_symbols)

            results = results.drop(columns=["speaker_answer_original"])
        elif "qwen" in file_name.lower() and "instruct" in file_name.lower():
            results["speaker_answer"] = results["speaker_answer"].apply(remove_symbols).strip()

        results["speaker_answer"] = results["speaker_answer"].str.lower()

        results["target_shape_list"] = results["target_shape"].str.split(r",\s*")
        results["target_texture_list"] = results["target_texture"].str.split(r",\s*")

        results[["contain_shape", "contain_color", "contain_texture"]] = results.apply(
            check_features, axis=1
        )
        results.to_csv(f"{file_name}_annotated.csv", index=False)

    elif task == "listener":
        target_location = results["target_location"].astype(str)
        listener_answer = results["listener_answer"].apply(get_numbers)

        results["listener_answer_correct"] = target_location == listener_answer
        
        results.to_csv(f"{file_name}.csv", index=False)






