import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def llm_init(model_name):
    global model, tokenizer

    models = {
        "olmo32b": "allenai/OLMo-2-0325-32B-Instruct",
        "olmo7b": "allenai/OLMo-2-1124-7B-Instruct",
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
    }
    if model_name in models:
        base = models[model_name]
        model = AutoModelForCausalLM.from_pretrained(base).half()
        tokenizer = AutoTokenizer.from_pretrained(base)
        model.to(device)
    elif "honeybee" in model_name:
        base = "yahma/llama-7b-hf"
        base_model = AutoModelForCausalLM.from_pretrained(base).half()
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = PeftModel.from_pretrained(base_model, model_name)
        model.to(device)
    elif any(key in model_name for key in models):
        for key, base in models.items():
            if key in model_name:
                base_model = AutoModelForCausalLM.from_pretrained(base).half()
                tokenizer = AutoTokenizer.from_pretrained(base)
                model = PeftModel.from_pretrained(base_model, model_name)
                model.to(device)
                break
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).half()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)

def wipe_model():
    global model, tokenizer
    model, tokenizer = None, None
    del model
    del tokenizer

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def llm_response(
        prompt: str,
        model_name: str,
        probs: bool = False,
        temperature: float = 0.1,
        max_new_tokens: int = 200
    ):
    if "olmo" in model_name:
        messages = [{
            "role": "user", 
            "content": prompt
        }]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        outputs = model.generate(
            model_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            return_dict_in_generate=True,
            output_scores=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
            )
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        input_length = encodeds.shape[1]
        generated_ids = outputs.sequences[:, input_length:]

        token_probs = {}
        if probs:
            for tok, score in zip(generated_ids[0], transition_scores[0]):
                token_probs[tokenizer.decode(tok)] = np.exp(score.item())

        decoded = tokenizer.batch_decode(generated_ids)
        if probs:
            return decoded[0], token_probs
        else:
            return decoded[0]

    elif "llama" in model_name or "honeybee" in model_name:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

        input_length = input_ids.shape[1]
        generated_ids = outputs.sequences[:, input_length:]

        token_probs = {}
        if probs:
            for tok, score in zip(generated_ids[0], transition_scores[0]):
                token_probs[tokenizer.decode(tok)] = np.exp(score.item())

        decoded = tokenizer.batch_decode(generated_ids)
        if probs:
            return decoded[0], token_probs
        else:
            return decoded[0]