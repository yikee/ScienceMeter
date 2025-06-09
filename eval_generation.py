import os
import json
import random
import argparse
from time import sleep

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import AzureOpenAI

import lm_utils
import parsing

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version
)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-bm", "--basemodel", help='Base model to use, e.g., "llama"')
    argParser.add_argument("-m", "--model", help='Updated model to use, e.g., "llama", "it_trainqadoc"')
    argParser.add_argument("-d", "--domain", help='Domain to evaluate, e.g., "computer_science"')
    argParser.add_argument("-k", "--knowledge", help='Knowledge type: "prior", "new", or "future"')
    argParser.add_argument("-o", "--portion", type=float, default=1.0, help="Portion of the dataset to use (default: 1.0)")

    args = argParser.parse_args()
    basemodel = args.basemodel
    model = args.model
    domain = args.domain
    knowledge = args.knowledge
    portion = args.portion

    if model in {"llama", "olmo7b", "olmo32b"}:
        model_to_use = model
    else:
        model_to_use = f"models/{basemodel}/{domain}/{model}"
    lm_utils.llm_init(model_to_use)

    dataset_path = f"dataset/filtered_with_claims/{basemodel}/{domain}.jsonl"
    with open(dataset_path, "r") as f:
        data = json.load(f)

    portion = float(portion)
    data = data[:int(len(data) * portion)]

    output = []

    def load_prompt(file_path, basemodel):
        with open(file_path) as f:
            prompt = f.read().rstrip("\n")
        return prompt

    base_path = "./prompts"
    generation_by_title_prompt = load_prompt(f"{base_path}/generation_by_title.prompt", basemodel)
    generation_by_subject_prompt = load_prompt(f"{base_path}/generation_by_subject.prompt", basemodel)

    with open("prompts/subject.prompt") as f:
        subject_prompt = f.read().rstrip("\n")

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(1000))
    def get_subject(abstract):
        response = client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": subject_prompt},
                {"role": "user", "content": abstract}
            ]
        )
        return response.choices[0].message.content

    with open("prompts/generation_factual_accuracy.prompt") as f:
        factual_accuracy_prompt = f.read().rstrip("\n")

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(1000))
    def get_factual_accuracy(abstract, claim):
        response = client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": factual_accuracy_prompt},
                {"role": "user", "content": "Paper Abstract: " + abstract + "\nClaim: " + claim}
            ]
        )
        return response.choices[0].message.content

    def run_task(prompt, abstract):
        response = lm_utils.llm_response(prompt, model_to_use, max_new_tokens=100)
        response_factual_accuracy = get_factual_accuracy(abstract, response)
        answer = parsing.yes_no_parsing(response_factual_accuracy)
        factual_accuracy = "accurate" if answer == "yes" else "inaccurate"

        moreinfo_prompt = prompt + "\nDo you need more information to do this task? Answer Yes or No only.\nAnswer: "
        moreinfo_response = lm_utils.llm_response(moreinfo_prompt, model_to_use, max_new_tokens=10)
        moreinfo = parsing.yes_no_parsing(moreinfo_response)

        output.append({
            "paperId": paper["paperId"],
            "model_response": response,
            "factual_accuracy": factual_accuracy,
            "moreinfo": moreinfo,
        })

    for d in tqdm(data):
        if knowledge in {"prior", "new"}:
            paper = d[f"{knowledge}_paper"]
            prompt = generation_by_title_prompt.format(title=paper["title"])
            run_task(prompt, paper["abstract"])
        elif knowledge == "future":
            paper = d["future_paper"]
            subject = get_subject(paper["abstract"])
            prompt = generation_by_subject_prompt.format(subject=subject)
            run_task(prompt, paper["abstract"])

    folder_path = os.path.join("results", basemodel, domain, "generation")
    file_path = os.path.join(folder_path, f"{model}_{knowledge}.json")

    os.makedirs(folder_path, exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(output, file)

    print("-----------------")
    print("Base Model: ", basemodel)
    print("Model: ", model)
    print("Task: ", "Claim Generation Task")
    print("Domain: ", domain)
    print("Knowledge: ", knowledge)
    print("-----------------")