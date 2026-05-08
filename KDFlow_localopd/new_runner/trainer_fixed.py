
import os
import time
from datetime import timedelta
from typing import Dict, List, Any
from collections import defaultdict

import ray
import torch
import torch.distributed as dist

try:
    from kdflow.datasets.utils import get_tokenizer_or_processor
    from kdflow.utils.logging_utils import init_logger
    from kdflow.utils.utils import zero_pad_sequences
except Exception:
    def init_logger(name):
        class DummyLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
        return DummyLogger()

    def zero_pad_sequences(vals, side="right", value=0):
        from torch.nn.utils.rnn import pad_sequence
        return pad_sequence(vals, batch_first=True, padding_value=value)

    def get_tokenizer_or_processor(*args, **kwargs):
        return None

from new_runner.config import LocalOPDConfig
from new_runner.pipeline import compute_local_divergence_loss_for_trajectory


logger = init_logger(__name__)


class OnPolicyKDTrainer:


    def __init__(
        self,
        strategy,
        student_model,
        teacher_model,
        rollout_group,
        is_same_tokenizer: bool,
        train_dataloader,
        eval_dataloader=None,
        max_rollout_iters: int = None,
        num_rollout_iters_per_epoch: int = None,
        generate_kwargs: Dict[str, float] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            strategy: Training strategy containing configuration
            student_model: StudentActorGroup
            teacher_model: TeacherActorGroup
            rollout_group: RolloutGroup
            is_same_tokenizer: Whether student and teacher use same tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            max_rollout_iters: Maximum rollout iterations in training
            num_rollout_iters_per_epoch: Number of rollout iterations per epoch
        """
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher = teacher_model
        self.rollout_group = rollout_group
        self.is_same_tokenizer = is_same_tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_rollout_iters = max_rollout_iters
        self.num_rollout_iters_per_epoch = num_rollout_iters_per_epoch
        self.generate_kwargs = generate_kwargs or {}
        self.epochs = self.args.train.num_epochs

        self.image_key = getattr(self.args.data, "image_key", None)
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path,
            need_processor=self.image_key is not None,
        )

        self.teacher_processor = None
        if self.args.model.teacher_name_or_path:
            if not self.is_same_tokenizer:
                self.teacher_processor = get_tokenizer_or_processor(
                    self.args.model.teacher_name_or_path,
                    need_processor=self.image_key is not None,
                )
            else:
                self.teacher_processor = self.student_processor

        self.world_size = (
            self.args.train.num_nodes * self.args.train.num_gpus_per_node
        )

        assert (
            self.args.kd.kd_ratio == 1.0
        ), "On-policy KD only supports kd_ratio=1.0."

        self.log_state = defaultdict(list)
        self._init_loggers()

        # =========================
        # Local divergence-aware OPD config
        # =========================
        args = self.args
        self.local_opd_cfg = LocalOPDConfig(
            enabled=getattr(args, "enable_local_opd", False),

            # Step 0 / full rollout
            num_full_rollouts=getattr(args, "local_num_full_rollouts", 4),

            # Step 1 / candidate detection
            num_candidates=getattr(args, "local_num_candidates", 3),
            min_candidate_separation=getattr(args, "local_min_candidate_sep", 8),
            rollback_steps=getattr(args, "local_rollback_steps", 1),

            # Step 2 / cheap probe
            probe_len=getattr(args, "local_probe_len", 20),
            threshold_alpha=getattr(args, "local_threshold_alpha", 2.0),
            threshold_eps=getattr(args, "local_threshold_eps", 1e-6),

            # Step 3 / local continuation matching
            local_len=getattr(args, "local_cont_len", 20),
            teacher_num_samples=getattr(args, "local_teacher_samples", 4),
            student_num_samples=getattr(args, "local_student_samples", 4),
            temperature=getattr(args, "local_temperature", 1.0),
            top_p=getattr(args, "local_top_p", 1.0),

            # Step 4 / local loss
            distance_type=getattr(args, "local_distance_type", "l1"),
            local_loss_weight=getattr(args, "local_loss_weight", 1.0),

            debug=getattr(args, "local_debug", False),
        )

        self.student_tokenizer = getattr(self.student, "tokenizer", None)
        self.teacher_tokenizer = getattr(self.teacher, "tokenizer", None)

        self.pad_token_id = None
        self.eos_token_id = None
        if self.student_tokenizer is not None:
            self.pad_token_id = getattr(self.student_tokenizer, "pad_token_id", None)
            self.eos_token_id = getattr(self.student_tokenizer, "eos_token_id", None)

    def _init_loggers(self) -> None:
        """Initialize wandb loggers."""
        self._wandb = None

        if self.args.log.use_wandb:
            import wandb

            self._wandb = wandb
            if self.args.log.wandb_mode != "offline" and not wandb.api.api_key:
                wandb.login()
            wandb.init(
                entity=self.args.log.wandb_org,
                project=self.args.log.wandb_project,
                group=self.args.log.wandb_group,
                name=self.args.log.wandb_run_name,
                config=vars(self.args),
                reinit=True,
                mode=self.args.log.wandb_mode,
                dir=self.args.log.wandb_dir,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*",
                step_metric="train/global_step",
                step_sync=True,
            )
            wandb.define_metric("eval/global_step")
            wandb.define_metric(
                "eval/*",
                step_metric="eval/global_step",
                step_sync=True,
            )

    def _print_training_config(self) -> None:
        """Log training configuration before training starts."""
        total_steps = self.max_rollout_iters
        grad_accum = (
            self.args.train.train_batch_size * self.args.model.ring_attn_size
            // (
                self.args.train.micro_train_batch_size
                * self.args.train.num_nodes
                * self.args.train.num_gpus_per_node
            )
        )

        logger.info("******* Start Training *******")
        logger.info(f"  Num Epochs:            {self.epochs}")
        logger.info(f"  Steps per Epoch:       {self.num_rollout_iters_per_epoch}")
        logger.info(f"  Total Training Steps:  {total_steps}")
        logger.info(f"  Per-device Batch Size: {self.args.train.micro_train_batch_size}")
        logger.info(f"  Gradient Accumulation: {grad_accum}")
        logger.info(f"  Learning Rate:         {self.args.train.learning_rate}")
        logger.info(f"  KD Algorithm:          {self.args.kd.kd_algorithm}")
        logger.info(f"  KD Loss Function:      {self.args.kd.kd_loss_fn}")

    def fit(self, global_step=0, start_epoch=0):
        # get eval and save steps
        if self.args.train.eval_steps == -1:
            self.args.train.eval_steps = float("inf")  # Evaluate once per epoch
        if self.args.train.save_steps == -1:
            # do not save ckpt mid-epoch
            self.args.train.save_steps = self.num_rollout_iters_per_epoch

        self.global_step = global_step



        # Create Gloo IPC groups between training ranks and rollout engines
        rollout_tp_size = getattr(self.args.rollout, "rollout_tp_size", 1)
        self.student.connect_rollout_engines(
            self.rollout_group.actors,
            rollout_tp_size,
        )

        self.start_time = time.time()
        status = defaultdict(list)
        num_micro_batches = (
            self.args.train.train_batch_size
            // self.args.train.micro_train_batch_size
        )

        ALLOWED_KEYS = {
                        "tea_input_ids", "tea_attn_mask", "tea_loss_mask",
                        "stu_input_ids", "stu_attn_mask", "stu_loss_mask",
                        "tea_full_texts", "rollout_log_probs",
                        "stu_prompts", "stu_responses", "tea_prompts", "labels",
                        "response_length", "total_length",
                        "teacher_hiddens", "avg_micro_batch_token_num",
                    } 

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)

            for prompt_batch in self.train_dataloader:
                self.global_step += 1

                rollout_start = time.time()
                rollout_samples = self.rollout(prompt_batch, **self.generate_kwargs)
                rollout_time = time.time() - rollout_start
                self.log_state["rollout_time"].append(rollout_time)

                teacher_start = time.time()
                # if self.args.kd.teacher_enable_sleep:
                #     self.teacher.wakeup()
                if self.args.train.enable_sleep:
                    self.teacher.wakeup()

                rollout_samples_for_kd = self.teacher.forward(rollout_samples)

                # if self.args.kd.teacher_enable_sleep:
                #     self.teacher.sleep()
                if self.args.train.enable_sleep:
                    self.teacher.sleep()

                self.log_state["teacher_fwd_time"].append(
                    time.time() - teacher_start
                )

                all_global_batches = []
                for i in range(0, len(rollout_samples_for_kd), num_micro_batches):
                    global_batch = rollout_samples_for_kd[i : i + num_micro_batches]

                    if len(global_batch) == 0:
                        # self.strategy.print(
                        #     f"[debug] skip empty global_batch at slice "
                        #     f"{i}:{i + num_micro_batches}"
                        # )
                        continue

                    valid_global_batch = []
                    for mb_idx, mb in enumerate(global_batch):
                        if "stu_loss_mask" not in mb:
                            # self.strategy.print(
                            #     "[debug] skip micro-batch without stu_loss_mask "
                            #     f"at slice {i}, mb_idx={mb_idx}, "
                            #     f"keys={list(mb.keys())}"
                            # )
                            continue

                        token_num = mb["stu_loss_mask"].sum()
                        try:
                            token_num_value = token_num.item()
                        except Exception:
                            token_num_value = float(token_num)

                        if token_num_value <= 0:
                            # self.strategy.print(
                            #     "[debug] skip micro-batch with empty "
                            #     f"stu_loss_mask at slice {i}, mb_idx={mb_idx}"
                            # )
                            continue

                        valid_global_batch.append(mb)

                    if len(valid_global_batch) == 0:
                        # self.strategy.print(
                        #     "[debug] skip global_batch because all micro-batches "
                        #     f"are invalid at slice {i}:{i + num_micro_batches}"
                        # )
                        continue

                    global_batch_token_num = sum(
                        mb["stu_loss_mask"].sum() for mb in valid_global_batch
                    )
                    avg_micro_batch_token_num = (
                        global_batch_token_num / len(valid_global_batch)
                    )

                    for mb in valid_global_batch:
                        mb["avg_micro_batch_token_num"] = avg_micro_batch_token_num

                    # self.strategy.print(
                    #     "[debug] append global_batch: "
                    #     f"num_micro_batches={len(valid_global_batch)}, "
                    #     f"token_sum={global_batch_token_num}"
                    # )
                    all_global_batches.append(valid_global_batch)

                # self.strategy.print(
                #     f"[debug] total valid global batches = {len(all_global_batches)}"
                # )

                student_start = time.time()

                # if self.args.train.train_enable_sleep:
                #     self.student.wakeup()
                if self.args.train.enable_sleep:
                    self.student.wakeup()

                if len(all_global_batches) == 0:
                    raise RuntimeError(
                        "No valid global batches were produced after teacher "
                        "forward. Likely causes: rollout returned empty samples, "
                        "all stu_loss_mask were empty, or samples were filtered "
                        "out before student training."
                    )

                for batch_idx, global_batch in enumerate(all_global_batches):
                    # self.strategy.print(
                    #     f"[debug] dispatch global_batch[{batch_idx}] with "
                    #     f"{len(global_batch)} micro-batches"
                    # )
                    
                    
                    filtered_batch = [
                        {k: v for k, v in mb.items() if k in ALLOWED_KEYS}
                        for mb in global_batch
                    ]

                    global_batch = filtered_batch                
                    status_list = ray.get(
                        self.student.async_run_distill(global_batch)
                    )
                    for k in status_list[0].keys():
                        self.log_state[k].append(
                            sum(s[k] for s in status_list) / len(status_list)
                        )

                self.log_state["student_train_time"].append(
                    time.time() - student_start
                )

                # if self.args.train.train_enable_sleep:
                #     self.student.sleep()

                if self.args.train.enable_sleep:
                    self.student.wakeup()

                if self.args.rollout.rollout_enable_sleep:
                    self.rollout_group.wakeup(tags=["weights"])
                # if self.args.train.enable_sleep:
                #     self.rollout_group.wakeup(tags=["weights"])

                update_start = time.time()
                self.student.update_rollout_weights()
                self.log_state["weight_update_time"].append(
                    time.time() - update_start
                )

                if self.args.rollout.rollout_enable_sleep:
                    self.rollout_group.sleep(tags=["weights"])
                # if self.args.train.enable_sleep:
                #     self.rollout_group.wakeup(tags=["weights"])

                self.logging()

            # save model after each epoch
            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(
                self.args.train.save_path,
                f"epoch_{epoch + 1}",
            )
            ray.get(self.student.async_save_model(save_path))

        total_time = time.time() - self.start_time
        self.strategy.log(
            "Training done, totally cost "
            f"{str(timedelta(seconds=total_time)).split('.')[0]}"
        )

        if self._wandb is not None:
            self._wandb.finish()

    def rollout(self, prompt_batch: List[Dict[str, str]], **kwargs) -> List[dict]:
        """Generate samples using rollout engine.

        Args:
            prompt_batch: List of dicts with keys:
                datasource, stu_prompt, tea_prompt, label
            **kwargs: Additional arguments for generation

        Returns:
            List of rollout micro-batch dicts containing generated samples
        """
        if self.args.rollout.rollout_enable_sleep:
            self.rollout_group.wakeup()

        # Extract prompts and labels from batch
        all_stu_prompts = [item["stu_prompt"] for item in prompt_batch]
        all_tea_prompts = [item["tea_prompt"] for item in prompt_batch]
        all_labels = [item["label"] for item in prompt_batch]
        all_images = (
            [item.get("images") for item in prompt_batch]
            if self.image_key
            else None
        )

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = self.args.rollout.n_samples_per_prompt
        all_stu_prompts = sum(
            [[p] * n_samples_per_prompt for p in all_stu_prompts],
            [],
        )
        all_tea_prompts = sum(
            [[p] * n_samples_per_prompt for p in all_tea_prompts],
            [],
        )
        all_labels = sum(
            [[label] * n_samples_per_prompt for label in all_labels],
            [],
        )
        if all_images:
            all_images = sum(
                [[imgs] * n_samples_per_prompt for imgs in all_images],
                [],
            )

        all_outputs = self.rollout_group.generate(
            all_stu_prompts,
            self.generate_kwargs,
            image_data=all_images,
        )

        # Process outputs into rollout samples
        sample_list = []
        for i in range(len(all_outputs)):

            s = self._build_rollout_sample(
                stu_prompt=all_stu_prompts[i],
                tea_prompt=all_tea_prompts[i],
                output=all_outputs[i],
                label=all_labels[i],
                images=all_images[i] if all_images and all_images[i] else None,
            )
            if s is not None:
                sample_list.append(s)

            
            
        



        # 🔥 关键修复：过滤所有非法 sample
        filtered_samples = []
       
        for s in sample_list:
            try:
                if s is None:
                    continue

                # === student检查 ===
                # if s["stu_response_ids"].numel() == 0:
                #     print("[DROP] stu_response_ids empty")
                #     continue
                # if s["stu_input_ids"].numel() == 0:
                #     print("[DROP] stu_input_ids empty")
                #     continue
                # if s["stu_loss_mask"].sum() == 0:
                #     print("[DROP] stu_loss_mask all zero")
                #     continue

                # # === 🔥新增 teacher检查（关键）===
                # if s["tea_response_ids"].numel() == 0:
                #     print("[DROP] tea_response_ids empty")
                #     continue
                # if s["tea_input_ids"].numel() == 0:
                #     print("[DROP] tea_input_ids empty")
                #     continue
                # if s["tea_loss_mask"].sum() == 0:
                #     print("[DROP] tea_loss_mask all zero")
                #     continue

                filtered_samples.append(s)

            except Exception:
                continue

        sample_list = filtered_samples

        # ===== 补充：过滤所有空字段 =====
        clean_samples = []
        for idx, s in enumerate(sample_list):
            bad = False
            for k, v in s.items():
                if isinstance(v, torch.Tensor) and v.numel() == 0:
                    # print(f"[DROP] sample {idx} key {k} empty tensor")
                    bad = True
                    break
                if isinstance(v, list) and len(v) == 0:
                    # print(f"[DROP] sample {idx} key {k} empty list")
                    bad = True
                    break
            if not bad:
                clean_samples.append(s)

        sample_list = clean_samples

        print(f"[ROLLOUT] final sample_list size = {len(sample_list)}")
        for idx, s in enumerate(sample_list[:3]):
            # print(f"[SAMPLE {idx}] keys={list(s.keys())}")
            for k, v in s.items():
                # if isinstance(v, torch.Tensor):
                #     print(f"  {k}: shape={tuple(v.shape)}, dim={v.dim()}, numel={v.numel()}")
                # elif isinstance(v, list):
                #     print(f"  {k}: list len={len(v)}")
                # else:
                #     print(f"  {k}: type={type(v)}")

                micro_batch_list = self._collate_micro_batches(
                    sample_list,
                    self.args.train.micro_train_batch_size,
                )

        if self.args.rollout.rollout_enable_sleep:
            self.rollout_group.sleep()

        return micro_batch_list

    @staticmethod
   
    def _collate_values(key: str, values: list):
        if len(values) == 0:
            print(f"[FATAL] empty values for key={key}")
            return None

        # print(f"\n[COLLATE] key={key}")
        # print("types:", [type(v) for v in values])

        v0 = values[0]

        if isinstance(v0, torch.Tensor):
            vals = [v if v.dim() > 0 else v.unsqueeze(0) for v in values]

            # 标量/长度字段：直接 stack，不走 pad_sequence
            if all(v.numel() == 1 for v in vals):
                return torch.stack([v.view(1) for v in vals], dim=0)

            if key.startswith("mm_"):
                return torch.cat(vals, dim=0)

            return zero_pad_sequences(vals, side="right", value=0)
            

        if isinstance(v0, list):
            for i, v in enumerate(values):
                print(f"  list[{i}] len={len(v)}")
                if len(v) == 0:
                    # print(f"[DROP-IN-COLLATE] key={key} has empty list at idx={i}")
                    return None
            return sum(values, [])

        if v0 is None:
            # print(f"[COLLATE] key={key} is None")
            return None

        # print(f"[COLLATE] key={key} fallback scalar values={values}")
        return values

    def _collate_micro_batches(
        self,
        sample_list: List[Dict],
        batch_size: int,
    ) -> List[Dict]:
        micro_batch_list = []
        for i in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[i : i + batch_size]

            micro_batch = {}
            for key in batch_samples[0]:
                vals = [s[key] for s in batch_samples]
                out = self._collate_values(key, vals)

                if out is None:
                    # print(f"[SKIP-KEY] key={key} returned None")
                    continue

                micro_batch[key] = out

            # print(f"[MICRO-BATCH] built keys={list(micro_batch.keys())}")
            micro_batch_list.append(micro_batch)

        return micro_batch_list

    def _tokenize_sample(
        self,
        prompt: str,
        response: str,
        processor,
        prefix: str,
        images=None,
    ) -> Dict[str, Any]:
        """Tokenize prompt + response for a single sample."""
        prompt_input = {"text": prompt}
        if images:
            prompt_input["images"] = images
        prompt_tok = processor(
            **prompt_input,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = prompt_tok["input_ids"].shape[1]

        full_input = {"text": prompt + response}
        if images:
            full_input["images"] = images

        # concat before tokenize to avoid tokenization mismatch
        full_tok = processor(
            **full_input,
            return_tensors="pt",
            add_special_tokens=False,
        )
        resp_len = full_tok["input_ids"].shape[1] - prompt_len

        input_ids = full_tok["input_ids"][0]
        attn_mask = full_tok["attention_mask"][0]
        response_ids = input_ids[prompt_len:]
        loss_mask = torch.tensor(
            [False] * (prompt_len - 1) + [True] * (resp_len + 1)
        )

        result = {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attn_mask": attn_mask,
            f"{prefix}_loss_mask": loss_mask,
            f"{prefix}_prompt_ids": prompt_tok["input_ids"][0],
            f"{prefix}_response_ids": response_ids,
            f"{prefix}_prompt_len": torch.tensor([prompt_len], dtype=torch.long),
        }

        # Extract multimodal fields (e.g., pixel_values, image_grid_thw)
        for k, v in prompt_tok.items():
            if k not in ("input_ids", "attention_mask"):
                v = torch.as_tensor(v)
                result[f"mm_{k}"] = v.squeeze(0) if v.dim() > 2 else v

        return result

    def _build_rollout_sample(
        self,
        stu_prompt: str,
        tea_prompt: str,
        output,
        label: str,
        images=None,
    ) -> Dict[str, Any]:
        """
        Build a single rollout sample with both student and teacher tokenizations.
        """
        # Decode response using student tokenizer
        response_ids = output["output_ids"]
        if len(response_ids) == 0:
            # skip this sample
            return None
        response_text = output["text"]

        stu_tokens = self._tokenize_sample(
            stu_prompt,
            response_text,
            self.student_processor,
            "stu",
            images=images,
        )

        if not self.is_same_tokenizer or tea_prompt != stu_prompt:
            tea_tokens = self._tokenize_sample(
                tea_prompt,
                response_text,
                self.teacher_processor,
                "tea",
                images=images,
            )
        else:
            # Same tokenizer: reuse student tensors
            tea_tokens = {
                "tea_input_ids": stu_tokens["stu_input_ids"].clone(),
                "tea_attn_mask": stu_tokens["stu_attn_mask"].clone(),
                "tea_loss_mask": stu_tokens["stu_loss_mask"].clone(),
                "tea_prompt_ids": stu_tokens["stu_prompt_ids"].clone(),
                "tea_response_ids": stu_tokens["stu_response_ids"].clone(),
                "tea_prompt_len": stu_tokens["stu_prompt_len"].clone(),
            }

        response_length = len(response_ids)
        total_length = stu_tokens["stu_attn_mask"].float().sum()

        # Build tea_full_text for teacher actor (SGLang engine uses raw text)
        tea_full_text = tea_prompt + response_text

        sample = {
            **tea_tokens,
            **stu_tokens,
            "tea_full_texts": [tea_full_text],
            "rollout_log_probs": None,
            "stu_prompts": [stu_prompt],
            "stu_responses": [response_text],
            "tea_prompts": [tea_prompt],
            "labels": [label],
            "response_length": torch.FloatTensor([[response_length]]),
            "total_length": torch.FloatTensor([[total_length]]),

            # local-OPD friendly aliases
            "student_prompt_ids": stu_tokens["stu_prompt_ids"].clone(),
            "student_response_ids": stu_tokens["stu_response_ids"].clone(),
            "teacher_prompt_ids": tea_tokens["tea_prompt_ids"].clone(),
            "teacher_response_ids": tea_tokens["tea_response_ids"].clone(),
            "rollout_output_ids": torch.as_tensor(response_ids, dtype=torch.long),
        }
        if images:
            sample["images"] = [images]
        return sample

    def logging(self):
        if self.global_step % self.args.log.logging_steps == 0:
            progress = (
                self.global_step
                / self.num_rollout_iters_per_epoch
                / self.epochs
            )
            eta = int(time.time() - self.start_time) * (1 - progress) / progress
            progress_str = (
                "epoch [{current_epoch}/{total_epoch}], "
                "step [{current_step}/{total_step}], "
                "train_progress [{progress:.2f}%], "
                "Elapsed: {elapsed}, "
                "ETA: {eta}, "
            ).format(
                current_epoch=self.current_epoch + 1,
                total_epoch=self.epochs,
                current_step=self.global_step,
                total_step=self.num_rollout_iters_per_epoch * self.epochs,
                progress=progress * 100,
                elapsed=str(
                    timedelta(seconds=(time.time() - self.start_time))
                ).split(".")[0],
                eta=str(timedelta(seconds=eta)).split(".")[0],
            )
            for k in self.log_state:
                if isinstance(self.log_state[k], list) and len(self.log_state[k]) > 0:
                    self.log_state[k] = sum(self.log_state[k]) / len(self.log_state[k])

            log_info = []
            for k in self.log_state:
                if k == "lr":
                    log_info.append(f"lr: {self.log_state[k]:.6e}")
                else:
                    log_info.append(f"{k}: {self.log_state[k]:.6f}")

            log_str = progress_str + ", ".join(log_info)
            self.strategy.log(log_str)

            if self._wandb is not None:
                logs = {"train/global_step": self.global_step}
                for k in self.log_state:
                    logs[f"train/{k}"] = self.log_state[k]
                self._wandb.log(logs)

            for k in self.log_state:
                self.log_state[k] = []

    # ============================================================
    # Local-OPD helper methods
    # ============================================================

    def _get_response_token_ids(self, sample):
        """
        Return the student response token ids for one sample.
        Shape: [T]
        """
        if "student_response_ids" in sample:
            return sample["student_response_ids"]
        if "stu_response_ids" in sample:
            return sample["stu_response_ids"]
        if "response_token_ids" in sample:
            return sample["response_token_ids"]
        if "student_token_ids" in sample:
            return sample["student_token_ids"]
        raise KeyError(
            "Cannot find response token ids in sample. "
            "Please map the actual field name here."
        )

    def _get_prompt_token_ids(self, sample):
        """
        Return prompt token ids for one sample.
        Shape: [P]
        """
        if "student_prompt_ids" in sample:
            return sample["student_prompt_ids"]
        if "stu_prompt_ids" in sample:
            return sample["stu_prompt_ids"]
        if "prompt_ids" in sample:
            return sample["prompt_ids"]
        if "input_ids" in sample:
            # only valid if input_ids means prompt-only in your actual data path
            return sample["input_ids"]
        raise KeyError(
            "Cannot find prompt token ids in sample. "
            "Please map the actual field name here."
        )

    def build_prefix_from_sample(self, sample, prefix_idx: int) -> torch.Tensor:
        """
        Build rollback prefix from:
            prompt + response[:prefix_idx]

        prefix_idx is relative to the response sequence, NOT absolute position
        in prompt+response.
        """
        prompt_ids = self._get_prompt_token_ids(sample)
        response_ids = self._get_response_token_ids(sample)

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        if isinstance(response_ids, list):
            response_ids = torch.tensor(response_ids, dtype=torch.long)

        prompt_ids = prompt_ids.view(-1)
        response_ids = response_ids.view(-1)

        prefix_idx = max(0, min(prefix_idx, response_ids.size(0)))
        prefix_response = response_ids[:prefix_idx]
        prefix_ids = torch.cat([prompt_ids, prefix_response], dim=0)
        return prefix_ids

    def extract_original_suffix(
        self,
        sample,
        prefix_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        """
        Extract local suffix from the original student full response.
        """
        response_ids = self._get_response_token_ids(sample)
        if isinstance(response_ids, list):
            response_ids = torch.tensor(response_ids, dtype=torch.long)

        response_ids = response_ids.view(-1)
        start = max(0, prefix_idx)
        end = min(response_ids.size(0), start + max_len)
        return response_ids[start:end]

    def get_token_losses_from_sample(self, sample) -> torch.Tensor:
        """
        Return token-level student losses aligned with response tokens.

        NOTE:
        This is still a placeholder until you expose token-level student loss
        from your rollout / teacher-forward / student-forward path.
        """
        if "student_token_losses" in sample:
            x = sample["student_token_losses"]
        elif "token_losses" in sample:
            x = sample["token_losses"]
        else:
            raise KeyError(
                "Cannot find token-level losses in sample. "
                "Please expose them from your rollout / scoring pipeline."
            )

        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)
        return x

    @torch.no_grad()
    def run_probe_gap(
        self,
        sample,
        prefix_start_idx: int,
        candidate_idx: int,
    ) -> float:
        """
        Compute cheap probe gap on the original student suffix.

        Current practical version:
          - build rollback prefix
          - take original student suffix of length probe_len
          - score the SAME suffix with teacher and student
          - return avg(log p_T - log p_S)
        """
        del candidate_idx  # kept for future use / debugging

        prefix_input_ids = self.build_prefix_from_sample(sample, prefix_start_idx)
        suffix_ids = self.extract_original_suffix(
            sample=sample,
            prefix_idx=prefix_start_idx,
            max_len=self.local_opd_cfg.probe_len,
        )

        if suffix_ids.numel() == 0:
            return float("-inf")

        score_out = self.score_continuation_teacher_vs_student(
            prefix_input_ids=prefix_input_ids,
            continuation_token_ids=suffix_ids,
        )

        teacher_logprobs = score_out["teacher_logprobs"]
        student_logprobs = score_out["student_logprobs"]

        gap = (teacher_logprobs - student_logprobs).mean()
        return float(gap.item())

    @torch.no_grad()
    def sample_teacher_local(
        self,
        prefix_input_ids,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Sample local continuations from teacher given a rollback prefix.
        """
        if num_samples <= 0:
            return []

        outputs = self._generate_from_model(
            model_group=self.teacher,
            prefix_input_ids=prefix_input_ids,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return [
            {"token_ids": out_ids, "source": "teacher_sample"}
            for out_ids in outputs
        ]

    @torch.no_grad()
    def sample_student_local(
        self,
        prefix_input_ids,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Sample local continuations from student given a rollback prefix.
        """
        if num_samples <= 0:
            return []

        outputs = self._generate_from_model(
            model_group=self.student,
            prefix_input_ids=prefix_input_ids,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return [
            {"token_ids": out_ids, "source": "student_sample"}
            for out_ids in outputs
        ]

    @torch.no_grad()
    def _generate_from_model(
        self,
        model_group,
        prefix_input_ids,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Wrapper for local continuation generation.

        TODO:
          Replace this with your actual KDFlow actor-group generation API.

        Expected return:
          list[Tensor], each Tensor is ONLY the newly generated continuation ids.
        """
        if isinstance(prefix_input_ids, list):
            prefix_input_ids = torch.tensor(prefix_input_ids, dtype=torch.long)

        prefix_input_ids = prefix_input_ids.view(-1).unsqueeze(0)  # [1, P]

        raise NotImplementedError(
            "Please map _generate_from_model() to the actual "
            "teacher/student generation API in your KDFlow setup."
        )

    def score_continuation_teacher_vs_student(
        self,
        prefix_input_ids,
        continuation_token_ids,
    ):
        """
        Score the SAME continuation under teacher and student.

        Returns:
        {
            "teacher_logprobs": Tensor[L],
            "student_logprobs": Tensor[L],
        }
        """
        teacher_logprobs = self._score_with_model(
            model_group=self.teacher,
            prefix_input_ids=prefix_input_ids,
            continuation_token_ids=continuation_token_ids,
            requires_grad=False,
        )

        student_logprobs = self._score_with_model(
            model_group=self.student,
            prefix_input_ids=prefix_input_ids,
            continuation_token_ids=continuation_token_ids,
            requires_grad=True,
        )

        return {
            "teacher_logprobs": teacher_logprobs,
            "student_logprobs": student_logprobs,
        }

    def _score_with_model(
        self,
        model_group,
        prefix_input_ids,
        continuation_token_ids,
        requires_grad: bool,
    ):
        """
        Return token-level logprobs of continuation conditioned on prefix.

        TODO:
          Replace the forward call below with the real KDFlow actor/model API.
        """
        if isinstance(prefix_input_ids, list):
            prefix_input_ids = torch.tensor(prefix_input_ids, dtype=torch.long)
        if isinstance(continuation_token_ids, list):
            continuation_token_ids = torch.tensor(
                continuation_token_ids,
                dtype=torch.long,
            )

        prefix_input_ids = prefix_input_ids.view(-1)
        continuation_token_ids = continuation_token_ids.view(-1)

        input_ids = torch.cat(
            [prefix_input_ids, continuation_token_ids],
            dim=0,
        ).unsqueeze(0)
        prefix_len = prefix_input_ids.size(0)
        cont_len = continuation_token_ids.size(0)

        if requires_grad:
            outputs = model_group.model(input_ids=input_ids)
        else:
            with torch.no_grad():
                outputs = model_group.model(input_ids=input_ids)

        logits = outputs.logits
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)

        target_ids = input_ids[:, 1:]
        gathered = logprobs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        start = prefix_len - 1
        end = start + cont_len
        cont_logprobs = gathered[0, start:end]
        return cont_logprobs

    def maybe_compute_local_opd_loss(self, sample):
        """
        Optional helper for later integration.

        This is NOT called from fit() yet. It is here so you have a clean place
        to plug Local-OPD in once token-level losses and generation/scoring are
        fully wired to the current KDFlow training path.
        """
        result = compute_local_divergence_loss_for_trajectory(
            cfg=self.local_opd_cfg,
            full_student_token_ids=self._get_response_token_ids(sample),
            token_losses=self.get_token_losses_from_sample(sample),
            build_prefix_fn=lambda prefix_idx: self.build_prefix_from_sample(
                sample,
                prefix_idx,
            ),
            extract_original_suffix_fn=lambda prefix_idx, max_len: (
                self.extract_original_suffix(sample, prefix_idx, max_len)
            ),
            probe_fn=lambda prefix_start_idx, candidate_idx: self.run_probe_gap(
                sample=sample,
                prefix_start_idx=prefix_start_idx,
                candidate_idx=candidate_idx,
            ),
            teacher_sample_fn=self.sample_teacher_local,
            student_sample_fn=self.sample_student_local,
            score_fn=self.score_continuation_teacher_vs_student,
        )
        return result
