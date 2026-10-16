CFG=configs/robotwin_vlm_5tasks.yaml
PRETRAINED_MODEL=Pretrained_models/patchwam-robotwin/step_140000.pt
accelerate launch --multi_gpu --num_machines 1 --num_processes 4 --mixed_precision bf16 \
  -m patchwam.cli train --config $CFG \
  --initialize $PRETRAINED_MODEL \
  model.dit_lora=true \
  model.dit_lora_rank=64 model.dit_lora_alpha=128 model.dit_lora_dropout=0.05 \
  training.output_dir=runs/robotwin-vlm-dit-lora-5tasks