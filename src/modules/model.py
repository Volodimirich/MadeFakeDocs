from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer


def get_tokenizer(tokenizer_name):
    if tokenizer_name in ['gpt2', "ai-forever/rugpt3large_based_on_gpt2",
                          "ai-forever/rugpt3medium_based_on_gpt2"]:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:

            SPECIAL_TOKENS = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
            tokenizer.add_special_tokens(SPECIAL_TOKENS)
        tokenizer.padding_side = 'left'
        return tokenizer
    elif tokenizer_name in ["ai-forever/FRED-T5-1.7B", 'ai-forever/FRED-T5-large']:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name, eos_token='</s>')
        return tokenizer
    elif tokenizer_name in {'t5-v1_1-large', 't5-v1_1-small'}:
        return T5Tokenizer.from_pretrained(tokenizer_name)
    else:
        raise NotImplementedError(f"This tokenizer is not supported!")


def get_model(model_name, device, local_path='', is_local=False):
    if model_name in ['gpt2', "ai-forever/rugpt3large_based_on_gpt2",
                      "ai-forever/rugpt3medium_based_on_gpt2"]:
        model = local_path if is_local else model_name
        return GPT2LMHeadModel.from_pretrained(model).to(device)
    elif model_name in ["ai-forever/FRED-T5-1.7B", "ai-forever/FRED-T5-large"]:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model.to(device)
    elif model_name in {'t5-v1_1-large', 't5-v1_1-small'}:
        return T5ForConditionalGeneration.from_pretrained(f"google/{model_name}").to(device)
    else:
        raise NotImplementedError(f"This model is not supported!")
