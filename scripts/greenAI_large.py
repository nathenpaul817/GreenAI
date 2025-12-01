import os
import math
import torch
import warnings
from codecarbon import EmissionsTracker
from eco2ai import Tracker as EcoTracker
import eco2ai
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 1. SETUP: Global Tracker (CodeCarbon) ---
def setup_global_tracker():
    output_dir = "./emission_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize CodeCarbon for the MACRO view
    # Note: On local machines, this tracks your actual wall power usage if sensors are available
    tracker = EmissionsTracker(
        project_name="Tier_III_BERT_Global_Local",
        output_dir=output_dir,
        measure_power_secs=15,
        save_to_file=True
    )
    return tracker, output_dir

# --- 2. SETUP: Granular Tracker (Eco2AI) ---
def run_granular_training(trainer_instance):
    print(">>> [Eco2AI] Granular tracking started...")
    
    granular_tracker = eco2ai.Tracker(
        project_name="Tier_III_BERT_Granular_Local",
        file_name="eco2ai_granular_results.csv"
    )
    
    granular_tracker.start()
    trainer_instance.train()
    granular_tracker.stop()
    
    print(f">>> [Eco2AI] Granular tracking finished. Saved to eco2ai_granular_results.csv")

# --- 3. MAIN EXECUTION ---
def main():
    # Detect Hardware for your verification
    # This helps confirm if your M4 Mac  or NVIDIA GPU  is being used.
    if torch.cuda.is_available():
        print(f"--- HARDWARE DETECTED: NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)} ---")
    elif torch.backends.mps.is_available():
        print("--- HARDWARE DETECTED: Apple Silicon (MPS) ---")
    else:
        print("--- HARDWARE DETECTED: Standard CPU ---")

    cc_tracker, output_dir = setup_global_tracker()
    cc_tracker.start()

    try:
        print("--- Starting Experimental Workload: Tier III (Local) ---")

        print("Loading Dataset...")
        # WikiText-103 is large. If your internet is slow, this might take a moment.
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        
        # Slicing for testing purposes (remove .select() for full experiment)
        train_dataset = dataset['train'].select(range(1000))
        eval_dataset = dataset['validation'].select(range(100))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

        print("Tokenizing...")
        # 'num_proc' speeds up tokenization on multi-core local CPUs
        tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=4)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=4)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        # Determine if we should use FP16 (NVIDIA) or standard precision (Mac/CPU)
        # Note: M-series Macs sometimes prefer bf16, but fp16 is generally supported.
        use_fp16 = torch.cuda.is_available() 

        training_args = TrainingArguments(
            output_dir="./bert_wikitext_finetuned",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            weight_decay=0.01,
            fp16=use_fp16, # Only True if CUDA is available to prevent MPS errors
            logging_steps=50,
            use_mps_device=torch.backends.mps.is_available(), # Explicitly request MPS for Mac
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        run_granular_training(trainer)

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        emissions = cc_tracker.stop()
        print(f"--- Experiment Complete ---")
        print(f"1. Global Results: {output_dir}/emissions.csv")
        print(f"2. Granular Results: ./eco2ai_granular_results.csv")

if __name__ == "__main__":
    main()