import torch

from kdflow.loss import register_loss


@register_loss("rkl")
@torch.compile()
def compute_reverse_kl_div(
    student_logits,
    teacher_logits, 
    temperature=1.0,
    reduction="none",
    **kwargs
):
    
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits.to(student_logits.device)
    teacher_logits = teacher_logits / temperature

    student_log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    student_probs = student_log_probs.exp()

    rkl_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)
    
    if reduction == "mean":
        rkl_div = rkl_div.mean()
    elif reduction == "sum":
        rkl_div = rkl_div.sum()

    print("student_logits.device =", student_logits.device, flush=True)
    print("teacher_logits.device =", teacher_logits.device, flush=True)
    print("student_log_probs.device =", student_log_probs.device, flush=True)
    print("teacher_log_probs.device =", teacher_log_probs.device, flush=True)
    print("student_probs.device =", student_probs.device, flush=True)
    print("rkl_div.device =", rkl_div.device, flush=True)
        
    return rkl_div