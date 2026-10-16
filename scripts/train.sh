#!/usr/bin/env bash
set -euo pipefail

CFG=configs/robotwin_vlm_5tasks.yaml
PRETRAINED_MODEL=${PRETRAINED_MODEL:-Pretrained_models/patchwam-robotwin/step_140000.pt}
# Set RESUME_CHECKPOINT to a complete run directory to continue that run from
# its full optimizer, scheduler, RNG, and data cursor state. The launcher never
# passes both flags.
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

initialization=(--initialize "$PRETRAINED_MODEL")
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  initialization=(--resume "$RESUME_CHECKPOINT")
fi

accelerate launch --multi_gpu --num_machines 1 --num_processes 4 --mixed_precision bf16 \
  -m patchwam.cli train --config "$CFG" \
  "${initialization[@]}" \
  model.dit_lora=true \
  model.dit_lora_rank=64 model.dit_lora_alpha=128 model.dit_lora_dropout=0.05 \
  training.output_dir=runs/robotwin-vlm-dit-lora-5tasks
