from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from argparse import ArgumentParser

import torch

def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint directory.")
    args = parser.parse_args()
    
    device="cuda"
    model=AutoModelForCausalLM.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    tokenizer=AutoTokenizer.from_pretrained("google/gemma-3-27b-pt")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "The president"
    print(pipe(prompt, use_cache=True, max_new_tokens=10, temperature=0.01)[0]['generated_text'])
    print(pipe(prompt, use_cache=False, max_new_tokens=10, temperature=0.01)[0]['generated_text'])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # print("With use_cache")
    # outputs_cached = model.generate(input_ids, max_new_tokens=10, use_cache=True,  do_sample=False)
    # print(tokenizer.decode(outputs_cached[0], skip_special_tokens=True))
    # print("Without use_cache")
    # outputs_nocache = model.generate(input_ids, max_new_tokens=10, use_cache=False, do_sample=False)
    # print(tokenizer.decode(outputs_nocache[0], skip_special_tokens=True))


    prompt = "The president"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 1) First step: compute logits with and without cache
    with torch.no_grad():
        out_nocache = model(**inputs, use_cache=False)
        out_cache   = model(**inputs, use_cache=True)

    logits_nocache = out_nocache.logits[:, -1, :]
    logits_cache   = out_cache.logits[:, -1, :]
    print("step 0 max diff:", (logits_cache - logits_nocache).abs().max().item())

    # 2) Second step: feed one more token with and without using the cache
    next_token_nocache = logits_nocache.argmax(dim=-1)
    next_token_cache   = logits_cache.argmax(dim=-1)

    inputs2_nocache = tokenizer(prompt + tokenizer.decode(next_token_nocache), return_tensors="pt").to(device)
    inputs2_cache   = {"input_ids": next_token_cache.unsqueeze(0), "past_key_values": out_cache.past_key_values}

    with torch.no_grad():
        out2_nocache = model(**inputs2_nocache, use_cache=False)
        out2_cache   = model(**inputs2_cache, use_cache=True)

    logits2_nocache = out2_nocache.logits[:, -1, :]
    logits2_cache   = out2_cache.logits[:, -1, :]
    print("step 1 max diff:", (logits2_cache - logits2_nocache).abs().max().item())
main()