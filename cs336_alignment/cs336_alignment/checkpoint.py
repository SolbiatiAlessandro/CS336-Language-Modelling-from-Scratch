import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

def tokenize_prompt_and_output(prompt_str, output_str, tokenizer):
    input_ids, labels, response_mask = [], [], []
    max_len = -1e9
    for i, prompt in enumerate(prompt_str):
        current_ids = []
        tokens1 = tokenizer.encode(prompt)
        #current_ids.append(tokens1)
        tokens2 = tokenizer.encode(output_str[i])
        tokens_length = len(tokens1) + len(tokens2) - 1
        max_len = max(tokens_length, max_len)
        #current_ids.append(tokens2)
        #input_ids.append(current_ids)
    for i, prompt in enumerate(prompt_str):
        current_ids = []
        current_mask = []
        current_label = []
        tokens1 = tokenizer.encode(prompt)

        current_ids += tokens1
        #current_label += tokens1[1:]
        current_mask += [0] * (len(tokens1) - 1)

        tokens2 = tokenizer.encode(output_str[i])
        tokens_length = len(tokens1) + len(tokens2) 
        current_ids += tokens2
        #current_label += tokens2
        current_mask += [1] * len(tokens2)

        missing_length = (max_len + 1) - tokens_length
        current_ids += [tokenizer.pad_token_id] * missing_length
        current_label = current_ids[1:]
        current_ids = current_ids[:-1]
        current_mask += [0] * missing_length

        input_ids.append(torch.tensor(current_ids))
        response_mask.append(torch.tensor(current_mask))
        labels.append(torch.tensor(current_label))

    return {
            'input_ids': torch.stack(input_ids), 
            'labels': torch.stack(labels), 
            'response_mask': torch.stack(response_mask)}

def get_response_log_probs(
        model, input_ids, labels, return_token_entropy):
    #import pdb;pdb.set_trace()
    logits = model(input_ids)
    probs = F.softmax(logits.logits, dim=-1)
    logprobs = F.log_softmax(logits.logits, dim=-1)

    B, T = labels.shape
    _logprobs = logprobs.reshape(B * T, -1)
    labels = labels.reshape(-1)
    label_probs = _logprobs[torch.arange(B*T), labels].reshape(B, T)

    res = {'log_probs': label_probs}
    if return_token_entropy:
        entropy = (-1 * probs * logprobs).sum(dim=-1)
        res['token_entropy'] = entropy
    return res
