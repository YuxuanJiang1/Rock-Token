import torch
import torch.nn.functional as F
import json

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy




@register_algorithm("token_freeze_kd")
class TokenFreezeKD:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)

        token_freeze_path = getattr(self.args.kd, "token_freeze_path", None)
        self.freeze_weight = float(getattr(self.args.kd, "freeze_weight", 0.0))

        if token_freeze_path is not None:
            with open(token_freeze_path, "r") as f:
                token_ids = json.load(f)
            self.freeze_token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.freeze_token_ids = None


    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
    
    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for KD"

        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_hiddens = output["hidden_states"][-1][student_loss_mask]
        del output

        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)
        
        student_logits = self.student.model.lm_head(student_hiddens)
        minV = min(teacher_logits.shape[-1], student_logits.shape[-1])
        teacher_logits = teacher_logits[:, :minV]
        student_logits = student_logits[:, :minV]
        if teacher_logits.shape != student_logits.shape:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(self.args.model.student_model_path)
            _stu_tokens = _tokenizer.convert_ids_to_tokens(student_input_ids[student_loss_mask].cpu().tolist())
            _tea_tokens = _tokenizer.convert_ids_to_tokens(teacher_input_ids[teacher_loss_mask].cpu().tolist())
            assert False, \
                f"teacher: {teacher_logits.shape} vs student: {student_logits.shape}. " \
                f"student tokens: {_stu_tokens}, teacher tokens: {_tea_tokens}"
        
        token_loss = self.loss_fn(
            student_logits,
            teacher_logits,
            reduction="none",
        )

        # 防止 loss_fn 返回 [N, vocab] 或 [N, ...]
        if token_loss.dim() > 1:
            token_loss = token_loss.sum(dim=-1)

        student_token_ids = student_input_ids[student_loss_mask]

        if self.freeze_token_ids is not None:
            freeze_ids = self.freeze_token_ids.to(student_token_ids.device)
            freeze_mask = torch.isin(student_token_ids, freeze_ids)

            weights = torch.ones_like(token_loss)
            weights[freeze_mask] = self.freeze_weight

            kd_loss = (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

            if torch.distributed.get_rank() == 0:
                print(
                    f"[TokenFreezeKD] freeze hits: {freeze_mask.sum().item()} / {freeze_mask.numel()}",
                    flush=True,
                )
        else:
            kd_loss = token_loss.sum() / avg_token_num




        loss_info = {"loss": kd_loss, "kd_loss": kd_loss}
        
        if self.args.kd.kd_ratio < 1:
            student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits, student_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info