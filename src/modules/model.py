from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
import os


def get_tokenizer(tokenizer_name):
    if tokenizer_name in ['gpt2', "ai-forever/rugpt3large_based_on_gpt2",
                          "ai-forever/rugpt3medium_based_on_gpt2"]:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:

            SPECIAL_TOKENS = {'bos_token': '<bos>', 'eos_token': '<s>', 'pad_token': '<pad>', 'sep_token': '<sep>'}
            tokenizer.add_special_tokens(SPECIAL_TOKENS)
        tokenizer.padding_side = 'left'
        return tokenizer
    elif tokenizer_name in ["ai-forever/FRED-T5-1.7B", 'ai-forever/FRED-T5-large']:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name, eos_token='</s>')
        return tokenizer
    elif tokenizer_name in {'flan-t5-large', 'flan-t5-small'}:
        return T5Tokenizer.from_pretrained(f"google/{tokenizer_name}")
    else:
        raise NotImplementedError(f"This tokenizer is not supported!")


def get_model(model_name, device, local_path='', is_local=False):
    if model_name in ['gpt2', "ai-forever/rugpt3large_based_on_gpt2",
                      "ai-forever/rugpt3medium_based_on_gpt2"]:
        model = local_path if is_local else model_name
        return GPT2LMHeadModel.from_pretrained(local_path if is_local else model_name).to(device)
    elif model_name in ["ai-forever/FRED-T5-1.7B", "ai-forever/FRED-T5-large"]:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return model.to(device)
    elif model_name in {'flan-t5-large', 'flan-t5-small'}:
        return T5ForConditionalGeneration.from_pretrained(f"google/{model_name}").to(device)
    else:
        raise NotImplementedError(f"This model is not supported!")

# ЭТО НЕ БЫДЛОКОД, ЧЕСТНО. ЭТО ОПТИМАЛЬНЫЙ ЗАПУСК
def get_fred(message, device, model_folder):
    url = 'https://drive.google.com/drive/folders/1HJdnhfKA4jZw09_N7LbQp4De1nGAUdil?usp=sharing'
    if not os.path.exists(model_folder):
        gdown.download_folder(url, output=model_folder, quiet=False)
    tokenizer = GPT2Tokenizer.from_pretrained(model_folder, eos_token='</s>')

    model = T5ForConditionalGeneration.from_pretrained(model_folder).to(device)
    model.eval()

    def predict(text, model):
        inp = tokenizer(text, truncation=True, max_length=20, return_tensors='pt').input_ids.to(device)
        res = model.generate(
            input_ids=inp,
            max_length=750,
            do_sample=True,
            top_k=50,
            top_p=0.85,
        )
        return tokenizer.decode(res[0], skip_special_tokens=True)
    return predict(message, model)
