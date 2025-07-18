import torch
import time
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from bma import MixtureOfCausalLM

# ---- Config ----
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model1_path = "/nobackup/users/scarv/multi-teacher-distillation/le-carnet/checkpoints/english/model_weights"
model2_path = "/nobackup/users/scarv/multi-teacher-distillation/le-carnet/checkpoints/french/model_weights"
tokenizer_path = "/nobackup/users/scarv/multi-teacher-distillation/data/eng_fr_tokenizer/tokenizer.json"
eng_prompt = "Once upon a time, there was a curious child named Alex."
fr_prompt = "Il était une fois un enfant curieux nommé Alex."
max_length = 128
num_trials = 100

def main():
    # ---- Load models and tokenizer ----
    print("Loading models and tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]"
    })

    model1 = AutoModelForCausalLM.from_pretrained(model1_path).to(device)
    model2 = AutoModelForCausalLM.from_pretrained(model2_path).to(device)
    mixture_model = MixtureOfCausalLM(model1, model2).to(device)

    # ---- Prepare input ----
    eng_input_ids = tokenizer.encode(eng_prompt, return_tensors="pt").to(device)
    fr_input_ids = tokenizer.encode(fr_prompt, return_tensors="pt").to(device)

    # ---- Profile function ----
    def profile_model(model, name, input_ids, max_length, num_trials):
        torch.cuda.empty_cache()
        torch.cuda.synchronize() if device == "cuda" else None
        times = []
        for i in range(num_trials):
            start = time.time()
            with torch.no_grad():
                if name == "Mixture Model":
                    _ = model.generate(
                        input_ids,
                        max_length=max_length,
                        top_k=None,
                        temperature=0.7,
                        eos_token_id=None,  # Ignore EOS, always generate max_length
                    )
                else:
                    _ = model.generate(
                        input_ids,
                        max_length=max_length,
                        do_sample=True,
                        temperature=0.7,
                        eos_token_id=None,  # Ignore EOS, always generate max_length
                    )
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.time()
            times.append(end - start)
            if (i+1) % 10 == 0:
                print(f"{name}: Completed {i+1}/{num_trials}")
        avg_time = sum(times) / len(times)
        print(f"{name}: Average generation time per sample: {avg_time:.3f} seconds")
        return times

    # ---- Run profiling ----
    print("\nProfiling model 1 (English)...")
    profile_model(model1, "Model 1", eng_input_ids, max_length, num_trials)

    print("\nProfiling model 2 (French)...")
    profile_model(model2, "Model 2", fr_input_ids, max_length, num_trials)

    print("\nProfiling mixture model (BMA) with English prompt...")
    profile_model(mixture_model, "Mixture Model", eng_input_ids, max_length, num_trials)

    print("\nProfiling mixture model (BMA) with French prompt...")
    profile_model(mixture_model, "Mixture Model", fr_input_ids, max_length, num_trials)

if __name__ == "__main__":
    main()
    print("Profiling completed.")