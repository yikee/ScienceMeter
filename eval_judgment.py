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
    argParser.add_argument("-m", "--model", help='Language model to use, e.g., "llama", "it_trainqadoc"')
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

    name_map = {
            "llama": "llama3.1-8B",
            "olmo7b": "olmo2-7B",
            "olmo32b": "olmo2-32B",
            "honeybee": "honeybee",
        }
    model_key = name_map.get(basemodel, basemodel)
    dataset_path = f"dataset/filtered_with_claims/{model_key}/{domain}.jsonl"
    with open(dataset_path, "r") as f:
        data = json.load(f)

    portion = float(portion)
    data = data[:int(len(data) * portion)]

    output = []

    def load_prompt(file_path, basemodel):
        with open(file_path) as f:
            prompt = f.read().rstrip("\n")
        if basemodel == "llama":
            prompt += " Answer Yes or No only"
        elif basemodel != "honeybee":
            prompt += " Yes or No"
        return prompt

    verification_prompts = []
    classification_prompts = []
    base_path = "./prompts/honeybee" if basemodel == "honeybee" else "./prompts"
    for i in range(1, 4):
        verification_prompts.append(load_prompt(f"{base_path}/verification_v{i}.prompt", basemodel))
        classification_prompts.append(load_prompt(f"{base_path}/classification_v{i}.prompt", basemodel))

    with open("prompts/linguistic_confidence.prompt") as f:
        linguistic_confidence_prompt = f.read().rstrip("\n")

    with open("prompts/honeybee/verification_moreinfo.prompt") as f:
        honeybee_verification_moreinfo_prompt = f.read().rstrip("\n")
    with open("prompts/honeybee/classification_moreinfo.prompt") as f:
        honeybee_classification_moreinfo_prompt = f.read().rstrip("\n")

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(1000))
    def get_linguistic_confidence(model_response):
        response = client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": linguistic_confidence_prompt},
                {"role": "user", "content": model_response}
            ]
        )
        return response.choices[0].message.content

    def run_task(prompts, moreinfo_prompt, gold_answer):
        responses = []
        answers = []
        for prompt in prompts:
            response = lm_utils.llm_response(prompt, model_to_use)
            responses.append(response)
            answers.append(parsing.yes_no_parsing(response))

        factual_accuracy = "accurate" if answers[0] == gold_answer else "inaccurate"
        
        if answers == ["yes", "yes", "yes"] or answers == ["no", "no", "no"]:
            consistency = True
        else:
            consistency = False
        
        linguistic_confidence_response = get_linguistic_confidence(responses[0])
        linguistic_confidence = parsing.yes_no_parsing(linguistic_confidence_response)

        moreinfo_response = lm_utils.llm_response(moreinfo_prompt, model_to_use, max_new_tokens=10)
        moreinfo = parsing.yes_no_parsing(moreinfo_response)

        output.append({
            "paperId": paper["paperId"],
            "model_responses": responses,
            "model_choices": answers,
            "correct_choice": gold_answer,
            "factual_accuracy": factual_accuracy,
            "consistency": consistency,
            # "linguistic_confidence_response": linguistic_confidence_response,
            "linguistic_confidence": linguistic_confidence,
            "moreinfo": moreinfo,
        })

    for d in tqdm(data):
        if knowledge in {"prior", "new"}:
            paper = d[f"{knowledge}_paper"]
            for label, claim_key in [("yes", "synthetic_support_claim"), ("no", "synthetic_refute_claim")]:
                prompts = [
                    template.format(title=paper["title"], claim=paper[claim_key])
                    for template in verification_prompts
                ]
                if basemodel == "honeybee":
                    moreinfo_prompt = honeybee_verification_moreinfo_prompt.format(title=paper["title"], claim=paper[claim_key])
                else:
                    moreinfo_prompt = prompts[0] + "\nDo you need more information to answer this question? Answer Yes or No only.\nAnswer: "
                run_task(prompts, moreinfo_prompt, label)
        elif knowledge == "future":
            paper = d["future_paper"]
            for label, claim_key in [("yes", "synthetic_support_claim"), ("no", "synthetic_refute_claim")]:
                prompts = [
                    template.format(claim=paper[claim_key])
                    for template in classification_prompts
                ]
                if basemodel == "honeybee":
                    moreinfo_prompt = honeybee_classification_moreinfo_prompt.format(title=paper["title"], claim=paper[claim_key])
                else:
                    moreinfo_prompt = prompts[0] + "\nDo you need more information to answer this question? Answer Yes or No only.\nAnswer: "
                run_task(prompts, moreinfo_prompt, label)

    folder_path = os.path.join("results", basemodel, domain, "judgment")
    file_path = os.path.join(folder_path, f"{model}_{knowledge}.json")

    os.makedirs(folder_path, exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(output, file)

    print("-----------------")
    print("Base Model: ", basemodel)
    print("Model: ", model)
    print("Task: ", "Claim Judgment Task")
    print("Domain: ", domain)
    print("Knowledge: ", knowledge)
    print("-----------------")