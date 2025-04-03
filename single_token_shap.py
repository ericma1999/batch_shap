import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

# Define PAD token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id


def generate_masked_inputs(
    input_ids: torch.Tensor, num_masks: int, mask_prob: float = 0.3
):
    batch_size, seq_len = input_ids.shape
    input_ids = input_ids.expand(num_masks, seq_len)
    masks = torch.bernoulli((1 - mask_prob) * torch.ones(num_masks, seq_len)).to(device)
    masked_inputs = torch.where(
        masks.bool(), input_ids, pad_token_id * torch.ones_like(input_ids)
    )
    attention_masks = (masked_inputs != pad_token_id).long()
    return masked_inputs.to(device), attention_masks.to(device), masks.bool()


def model_forward_batched(input_ids_batch, attention_mask_batch, target_token_id):
    with torch.no_grad():
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits[:, -1, :]  # Next token logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs[:, target_token_id]


def compute_contribution_scores(
    input_ids, target_token_id, num_masks=64, mask_prob=0.3
):
    masked_inputs, attention_masks, masks = generate_masked_inputs(
        input_ids, num_masks, mask_prob
    )
    scores = model_forward_batched(
        masked_inputs, attention_masks, target_token_id
    )  # shape: (num_masks,)

    input_len = input_ids.shape[1]
    contributions = torch.zeros(input_len).to(device)

    for i in range(input_len):
        relevant = masks[:, i]
        if relevant.sum() > 0:
            avg_score = scores[relevant].mean()
            contributions[i] = avg_score

    norm = torch.sum(torch.abs(contributions))
    if norm > 0:
        contributions = contributions / norm
    return contributions.cpu().numpy()


def compute_cc_shap_score(pred_contribs, expl_contribs):
    min_len = min(len(pred_contribs), len(expl_contribs))
    pred_contribs = pred_contribs[:min_len]
    expl_contribs = expl_contribs[:min_len]
    return float(cosine_similarity([pred_contribs], [expl_contribs])[0][0])


if __name__ == "__main__":
    # 🧾 e-SNLI example
    premise = "A woman with a green headscarf, blue shirt and a very big grin."
    hypothesis = "The woman is very happy."
    label = " entailment"
    input_text = (
        f"Premise: {premise} Hypothesis: {hypothesis} What is the relationship?"
    )
    explanation_prompt = input_text + "\nExplain your answer."

    inputs_pred = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    input_ids_pred = inputs_pred["input_ids"]

    target_token_id = tokenizer.encode(label, add_special_tokens=False)[0]

    inputs_expl = tokenizer(explanation_prompt, return_tensors="pt", padding=True).to(
        device
    )
    input_ids_expl = inputs_expl["input_ids"]

    pred_contribs = compute_contribution_scores(input_ids_pred, target_token_id)

    expl_contribs = compute_contribution_scores(input_ids_expl, target_token_id)

    cc_shap_score = compute_cc_shap_score(pred_contribs, expl_contribs)
    print(f"CC-SHAP score: {cc_shap_score:.4f}")
