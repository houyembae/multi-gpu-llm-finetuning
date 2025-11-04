# Medical LLM Fine-Tuning

This project demonstrates multi-GPU fine-tuning of a Large Language Model (LLM) for medical dialogue processing.  
The model is adapted to generate structured SOAP (Subjective, Objective, Assessment, Plan) notes from doctor–patient conversations.

Key features include:
- Distributed training with DeepSpeed ZeRO-3
- Parameter-efficient fine-tuning using LoRA and QLoRA (4-bit)
- Configurable for single or multi-GPU environments

Full technical details are in Report.md
