from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint directory.")
    args = parser.parse_args()
    
    # test loading in megatron2huggingface
    # print("Loading model from:", args.checkpoint_path, "to megatron2huggingface")
    # model = megatron2huggingface.MegatronForCausalLM.from_pretrained(args.checkpoint_path)
    # print("Model loaded successfully in megatron2huggingface.")
    # test loading in transformers
    print("Loading model from:", args.checkpoint_path, "to transformers")
    model=AutoModelForCausalLM.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    print("Model loaded successfully in transformers.")
    tokenizer=AutoTokenizer.from_pretrained("google/gemma-3-27b-pt")
    model.generate(**tokenizer("Hello world", return_tensors="pt"))

main()