# DiT LoRA checkpoint

Status: accepted after review of the user-requested training-policy amendment.

Contract: vision and language encoders frozen; wrapped block Linear bases frozen and LoRA adapters trained; every other DiT parameter keeps the original full-training behavior. This includes txt_in, img_in, time_in, stream modulations, final_layer, and learnable QK normalization scales. State and optional Self-Flow projections remain fully trained. Full-mode behavior is unchanged.

Verification: root independently ran tests/test_dit_lora.py: 26 passed in 2.39s. Worker focused regression run: 108 passed and six preexisting accelerate resume failures, identical failure set to its pre-change run. Tests cover gradients, actual optimizer updates, encoder freezing, report inventory, and changed training-policy checkpoint signature. CPU synthetic models only; real-weight GPU behavior and end-to-end resume remain unverified.

Workspace: shared uncommitted changes; unrelated preexisting edits preserved. No commits or publishing. Worker /root/dit_lora finished and interrupted. No pending implementation tasks.

Routing: static doctor reported gpt-6.1-sol root and the DeepSeek API deepseek/deepseek-v4.1-flash role. Runtime provider metadata unavailable; inference routing unverified.
