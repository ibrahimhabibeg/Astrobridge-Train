from typing import List, Any
from . import EvalSample, ModelResponse
from ..tasks.distance_classification import DistanceClassPromptSpec, DistanceClassificationTask
from ..tasks.emission_lines import EmissionLinePromptSpec
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
        
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
        fusion_stack.load_state_dict(torch.load(middle_pt_path, map_location="cpu", weights_only=False))    
        print("Initializing full Captioner model and encoders...")
        self.model = Captioner(fusion_stack, llm, n_queries=int(self.cfg.qformer.n_queries))
        self.model.to(self.device)
        self.model.eval()
        self.encoders = {name: build_encoder(name, self.cfg.modalities[name], device=self.device) for name in self.cfg.modalities}
        self.max_tokens = {n: int(c.max_tokens) for n, c in self.cfg.modalities.items()}

    def get_config(self) -> dict:
        return {
            "type": "AstroBridgeResponder",
            "repo_id": self.repo_id
        }

    def _build_distance_prompt(self, spec: DistanceClassPromptSpec) -> str:
        return (
            "Based on the spectrum provided, classify the distance of the observed astronomical object into one of the following categories: "
            f"{spec.options_text}. "
            "Think step-by-step, but you MUST conclude with the exact phrase 'FINAL ANSWER: [Letter]'"
        )

    def _build_emission_prompt(self, spec: EmissionLinePromptSpec) -> str:
        # Instead of an instruction, we append a completion suffix to the base template.
        # This prevents the base LLM from falling back to chat/instruction priors
        # and generating <think> or assistant tokens.
        return " The visible emission lines in this spectrum are:"

    def respond_batch(self, samples: List[EvalSample], task: Any) -> List[ModelResponse]:
        if not hasattr(task, "get_prompt_spec") and hasattr(task, "format_options"):
            task = DistanceClassificationTask(task)

        spec = task.get_prompt_spec()
        
        question = None
        prompt_suffix = None
        
        if isinstance(spec, DistanceClassPromptSpec):
            # For distance classification, keep the zero-shot instruction
            question = self._build_distance_prompt(spec)
        elif isinstance(spec, EmissionLinePromptSpec):
            # For emission lines, we use the training template with a suffix
            prompt_suffix = self._build_emission_prompt(spec)
        else:
            question = task.default_prompt()

        raw_inputs_list = []
        for sample in samples:
            f_tensor = torch.tensor(sample.flux).float().unsqueeze(0)
            wavelength = torch.tensor(sample.wavelength).float().unsqueeze(0)
            
            spectrum_dict = {
                "flux": f_tensor,
                "wavelength": wavelength,
                "survey": [sample.survey]
            }
            
            if sample.ivar is not None:
                spectrum_dict["ivar"] = torch.tensor(sample.ivar).float().unsqueeze(0)
                
            if sample.mask is not None:
                spectrum_dict["mask"] = torch.tensor(sample.mask).bool().unsqueeze(0)
            else:
                spectrum_dict["mask"] = torch.zeros_like(f_tensor, dtype=torch.bool)
                
            raw_inputs_list.append({"spectra": spectrum_dict})
            
        answers = self._generate_caption_batch(raw_inputs_list, question=question, prompt_suffix=prompt_suffix)
        
        return [
            ModelResponse(
                parsed=task.default_parse(a),
                raw_text=a
            ) for a in answers
        ]

    def _generate_caption_batch(self, raw_inputs_list: list[dict], question: str = None, prompt_suffix: str = None) -> list[str]:        
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
        
            prompt_text = question if question is not None else self.cfg.prompt.template.format(modalities=human_readable_subset(shown))
            if prompt_suffix:
                prompt_text += prompt_suffix
            prompt_ids = self.tokenizer([prompt_text] * B, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        
            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                prefix = self.model.fusion_stack(modality_batch)
                prompt_embeds = self.model.llm.get_input_embeddings()(prompt_ids)
                inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
                
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                
                gen = self.model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
