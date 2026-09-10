from typing import List, Any

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel

from . import EvalSample, ModelResponse
from .fallback import identify_failed_indices, merge_fallback_responses
from captioner.utils.config import load_config
from captioner.encoders.registry import build_encoder
from captioner.model.captioner import Captioner, FusionStack
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.prompt import human_readable_subset


class AstroBridgeResponder:
    def __init__(self, config: dict, device: str):
        assert "astrobridge_id" in config and config["astrobridge_id"], "Missing 'astrobridge_id' in config"
        self.device = device
        self.repo_id = config["astrobridge_id"]
        self.cfg = load_config("base", "data", "modalities", "model", "stage2")

        print("Building base LLM...")
        llm, self.tokenizer = build_llm(self.cfg)
        print(f"Loading LoRA adapter from {self.repo_id}...")
        llm = PeftModel.from_pretrained(llm, self.repo_id)
        d_llm = get_llm_hidden_size(llm)

        print("Building FusionStack...")
        self.out_dims = {n: int(c.out_dim) for n, c in self.cfg.modalities.items()}
        fusion_stack = FusionStack(
            modality_out_dims=self.out_dims,
            d_shared=int(self.cfg.d_shared),
            d_llm=d_llm,
            qformer_cfg=dict(self.cfg.qformer),
            projector_hidden_mult=int(self.cfg.projector.hidden_mult),
            projector_dropout=float(self.cfg.projector.dropout),
        )
        print(f"Downloading middle.pt from {self.repo_id}...")
        middle_pt_path = hf_hub_download(repo_id=self.repo_id, filename="middle.pt")
        fusion_stack.load_state_dict(torch.load(middle_pt_path, map_location="cpu", weights_only=True))

        print("Initializing full Captioner model and encoders...")
        self.model = Captioner(fusion_stack, llm, n_queries=int(self.cfg.qformer.n_queries))
        self.model.to(self.device)
        self.model.eval()
        self.encoders = {
            name: build_encoder(name, self.cfg.modalities[name], device=self.device)
            for name in self.cfg.modalities
        }
        self.max_tokens = {n: int(c.max_tokens) for n, c in self.cfg.modalities.items()}
        self.generate_max_tokens = config.get("max_tokens", 256)
        self.fallback_max_tokens = config.get("fallback_max_tokens", 128)

    def get_config(self) -> dict:
        return {"type": "AstroBridgeResponder", "repo_id": self.repo_id}

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        question = task.build_prompt(image_mode=False)

        raw_inputs_list = []
        for sample in samples:
            f_tensor = torch.tensor(sample.flux).float().unsqueeze(0)
            wavelength = torch.tensor(sample.wavelength).float().unsqueeze(0)

            spectrum_dict = {
                "flux": f_tensor,
                "wavelength": wavelength,
                "survey": [sample.survey],
            }

            if sample.ivar is not None:
                spectrum_dict["ivar"] = torch.tensor(sample.ivar).float().unsqueeze(0)

            if sample.mask is not None:
                spectrum_dict["mask"] = torch.tensor(sample.mask).bool().unsqueeze(0)
            else:
                spectrum_dict["mask"] = torch.zeros_like(f_tensor, dtype=torch.bool)

            raw_inputs_list.append({"spectra": spectrum_dict})

        # First pass
        answers = self._generate_caption_batch(
            raw_inputs_list, questions=question, max_new_tokens=self.generate_max_tokens
        )

        responses = [
            ModelResponse(
                parsed=task.default_parse(a) if task.default_parse(a) is not None else [],
                raw_text=a,
                forced_fallback=False,
            )
            for a in answers
        ]

        # Fallback pass
        fallback_tag = task.fallback_tag()
        failed = identify_failed_indices(responses)

        if failed and fallback_tag:
            fb_inputs = [raw_inputs_list[i] for i in failed]
            fb_questions = [question + "\n" + answers[i] + fallback_tag for i in failed]
            fb_answers = self._generate_caption_batch(
                fb_inputs, questions=fb_questions, max_new_tokens=self.fallback_max_tokens
            )
            merge_fallback_responses(responses, failed, fb_answers, fallback_tag, task)

        return responses

    def _generate_caption_batch(
        self,
        raw_inputs_list: list[dict],
        questions: list[str] | str | None = None,
        max_new_tokens: int = 256,
    ) -> list[str]:
        with torch.no_grad():
            if not raw_inputs_list:
                raise ValueError("raw_inputs_list is empty")

            shown = frozenset(raw_inputs_list[0].keys())
            B = len(raw_inputs_list)

            modality_batch = {}
            for name, out_dim in self.out_dims.items():
                T_m = self.max_tokens[name]
                tokens = torch.zeros((B, T_m, out_dim), dtype=torch.float32, device=self.device)
                mask = torch.ones((B, T_m), dtype=torch.bool, device=self.device)

                for i, raw_inputs in enumerate(raw_inputs_list):
                    if name in raw_inputs:
                        raw_tokens = self.encoders[name].encode(raw_inputs[name]).to(torch.float32)
                        n = min(raw_tokens.shape[1], T_m)
                        tokens[i, :n] = raw_tokens[0, :n].to(self.device)
                        mask[i, :n] = False
                modality_batch[name] = {"tokens": tokens, "mask": mask}

            if questions is not None:
                prompt_texts = [questions] * B if isinstance(questions, str) else questions
            else:
                prompt_texts = [
                    self.cfg.prompt.template.format(modalities=human_readable_subset(shown))
                ] * B

            self.tokenizer.padding_side = "left"
            prompt_ids = self.tokenizer(
                prompt_texts, add_special_tokens=False, return_tensors="pt", padding=True
            )["input_ids"].to(self.device)

            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                prefix = self.model.fusion_stack(modality_batch)
                prompt_embeds = self.model.llm.get_input_embeddings()(prompt_ids)
                inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], dtype=torch.long, device=self.device
                )

                pad_token_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )

                gen = self.model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
