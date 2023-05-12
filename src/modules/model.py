
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'gpt2':
        return GPT2Tokenizer.from_pretrained('gpt2')
    elif tokenizer_name in ('t5_v1_1-large', 't5_v1_1-small'):
        return T5Tokenizer.from_pretrained("google/{}")


def get_model(model_name, device, local_path='', is_local=False):
    if model_name == 'gpt2':
        model = local_path if is_local else model_name
        return GPT2LMHeadModel.from_pretrained(model).to(device)
    elif model_name == ('t5_v1_1-large', 't5_v1_1-small'):
        return T5ForConditionalGeneration.from_pretrained(f"google/{model_name}").to(device)  
    
    