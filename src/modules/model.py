
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name in ['gpt2', "sberbank-ai/rugpt3large_based_on_gpt2"] :
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side= 'left'
        return tokenizer

    elif tokenizer_name in {'flan-t5-large', 'flan-t5-small'}:
        return T5Tokenizer.from_pretrained(f"google/{tokenizer_name}")
    else:
        raise NotImplementedError


def get_model(model_name, device, local_path='', is_local=False):
    if model_name  in ['gpt2', "sberbank-ai/rugpt3large_based_on_gpt2"]:
        model = local_path if is_local else model_name
        return GPT2LMHeadModel.from_pretrained(model).to(device)
    elif model_name in {'flan-t5-large', 'flan-t5-small'}:
        return T5ForConditionalGeneration.from_pretrained(f"google/{model_name}").to(device)  
    else:
        raise NotImplementedError
    