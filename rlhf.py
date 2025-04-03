import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id


def generate_masked_inputs(
    input_ids: torch.Tensor, num_masks: int, mask_prob: float = 0.3
):
    batch_size, seq_len = input_ids.shape
    expanded = input_ids.unsqueeze(1).expand(batch_size, num_masks, seq_len)
    expanded = expanded.reshape(-1, seq_len)  # [batch_size * num_masks, seq_len]

    masks = torch.bernoulli((1 - mask_prob) * torch.ones_like(expanded)).to(device)
    masked_inputs = torch.where(
        masks.bool(), expanded, pad_token_id * torch.ones_like(expanded)
    )
    attention_masks = (masked_inputs != pad_token_id).long()
    return (
        masked_inputs.to(device),
        attention_masks.to(device),
        masks.reshape(batch_size, num_masks, seq_len),
    )


def model_forward_batched(
    input_ids_batch, attention_mask_batch, target_token_ids, target_positions
):
    with torch.no_grad():
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        idx = torch.arange(len(target_token_ids), device=device)
        selected_logits = probs[idx, target_positions, target_token_ids]
        return selected_logits


def compute_contribution_scores(
    input_ids_batch, target_token_ids, target_positions, num_masks=64, mask_prob=0.3
):
    batch_size, seq_len = input_ids_batch.shape
    masked_inputs, attention_masks, masks = generate_masked_inputs(
        input_ids_batch, num_masks, mask_prob
    )

    token_ids = torch.tensor(target_token_ids, device=device).repeat_interleave(
        num_masks
    )
    pos_ids = torch.tensor(target_positions, device=device).repeat_interleave(num_masks)

    scores = model_forward_batched(masked_inputs, attention_masks, token_ids, pos_ids)
    scores = scores.view(batch_size, num_masks)
    masks = masks.to(device)  # shape: (batch_size, num_masks, seq_len)

    contribs_all = []
    for i in range(batch_size):
        contributions = torch.zeros(seq_len).to(device)
        for j in range(seq_len):
            relevant = masks[i, :, j].bool()
            if relevant.sum() > 0:
                avg_score = scores[i][relevant].mean()
                contributions[j] = avg_score

        norm = torch.sum(torch.abs(contributions))
        if norm > 0:
            contributions = contributions / norm
        contribs_all.append(contributions.cpu().numpy())

    return contribs_all


def compute_explanation_contributions(
    input_ids_batch, explanation_output_ids_batch, num_masks=64, mask_prob=0.3
):
    batch_size = input_ids_batch.shape[0]
    all_contribs = []
    for i in range(batch_size):
        input_ids = input_ids_batch[i].unsqueeze(0)
        explanation_output_ids = explanation_output_ids_batch[i].unsqueeze(0)
        input_len = input_ids.shape[1]
        total_contribs = torch.zeros(input_len).to(device)
        T = explanation_output_ids.shape[1]

        for t in range(T):
            target_token_id = explanation_output_ids[0, t].item()
            contribs_t = compute_contribution_scores(
                input_ids, [target_token_id], [input_len - 1]
            )[0]
            total_contribs += torch.tensor(contribs_t, device=device)

        avg_contribs = total_contribs / T
        all_contribs.append(avg_contribs.cpu().numpy())
    return all_contribs


def compute_cc_shap_score(pred_contribs_list, expl_contribs_list):
    scores = []
    for pred_contribs, expl_contribs in zip(pred_contribs_list, expl_contribs_list):
        min_len = min(len(pred_contribs), len(expl_contribs))
        pred_contribs = pred_contribs[:min_len]
        expl_contribs = expl_contribs[:min_len]
        score = float(cosine_similarity([pred_contribs], [expl_contribs])[0][0])
        scores.append(score)
    return scores


if __name__ == "__main__":
    # make this more beautiful later
    premises = [
        "A woman with a green headscarf, blue shirt and a very big grin.",
        "A man is playing a guitar on a crowded street.",
    ]
    hypotheses = ["The woman is very happy.", "The man is performing for an audience."]
    labels = [" entailment", " entailment"]

    input_texts = [
        f"Premise: {p} Hypothesis: {h} What is the relationship?"
        for p, h in zip(premises, hypotheses)
    ]
    explanation_prompts = [txt + "\nExplain your answer." for txt in input_texts]

    # need to truncate here for some reason
    inputs_pred = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    input_ids_pred = inputs_pred["input_ids"]
    attention_mask_pred = inputs_pred["attention_mask"]

    target_token_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    target_positions = [input_ids_pred.shape[1] - 1] * len(target_token_ids)

    inputs_expl = tokenizer(
        explanation_prompts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    input_ids_expl = inputs_expl["input_ids"]
    attention_mask_expl = inputs_expl["attention_mask"]

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids_expl,
            attention_mask=attention_mask_expl,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
        )
        explanation_output_ids = generated[:, input_ids_expl.shape[1]:]

    print(" attribution for prediction")
    pred_contribs = compute_contribution_scores(
        input_ids_pred, target_token_ids, target_positions
    )

    print("azttribution for explanation")
    expl_contribs = compute_explanation_contributions(
        input_ids_expl, explanation_output_ids
    )

    scores = compute_cc_shap_score(pred_contribs, expl_contribs)
    for i, s in enumerate(scores):
        print(f"Sample {i}: CC-SHAP = {s:.4f}")
