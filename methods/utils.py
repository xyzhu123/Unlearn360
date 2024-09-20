import torch

def max_entropy_loss(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return -1 * entropy.mean()

def lm_loss(
        logits,
        labels,
        vocab_size,
):
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)

    loss_f = torch.nn.CrossEntropyLoss()
    loss = loss_f(shift_logits, shift_labels.to(shift_logits.device))
    return loss
