import numpy as np
import shap
import torch

from main import model, tokenizer


def model_forward_batched(input_ids_batch, attention_mask_batch, target_token_id):
    with torch.no_grad():
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        # Get logits of the target token at its position
        last_logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(last_logits, dim=-1)
        return probs[:, target_token_id]


def predict_with_mask(mask_matrix, input_ids, baseline_token_id=tokenizer.pad_token_id):
    batch_inputs = []
    attention_masks = []

    for mask in mask_matrix:
        masked_input = torch.where(
            torch.tensor(mask).bool(),
            input_ids,
            torch.tensor(baseline_token_id).expand_as(input_ids),
        )
        attention_mask = (masked_input != baseline_token_id).long()
        batch_inputs.append(masked_input)
        attention_masks.append(attention_mask)

    batch_inputs = torch.stack(batch_inputs).cuda()
    attention_masks = torch.stack(attention_masks).cuda()

    return (
        model_forward_batched(batch_inputs, attention_masks, target_token_id)
        .cpu()
        .numpy()
    )
