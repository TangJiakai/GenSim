import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from datasets import load_dataset
from trl import (
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import wandb
import argparse

from utils.constants import *
from utils.utils import check_load_adapter, check_dirs

wandb.init(mode="disabled")
tqdm.pandas()


def parse_args() -> argparse.Namespace:
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_mode", type=str, help="sft or ppo", default="sft")
    parser.add_argument(
        "--llm_path", type=str, help="Path to the base LLM model", required=True
    )
    return parser.parse_args()


def copy_saves():
    os.system(f"cp -r {TMP_SAVE_DIR}/* {SAVE_DIR}/")
    os.system(f"rm -rf {TMP_SAVE_DIR}")
    print(f"Copied the trained model to {SAVE_DIR}")


class Tuner:
    def __init__(self, tuning_mode, llm_path):
        self.tuning_mode = tuning_mode
        self.llm_path = llm_path
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        check_dirs(llm_path)

    def train_func(self):
        if self.tuning_mode == "sft":
            self.sft_train()
        elif self.tuning_mode == "ppo":
            self.ppo_train()
        copy_saves()

    def formatting_prompts_func(self, example):
        output_texts = []
        for i in range(len(example["prompt"])):
            text = f"### Question: {example['prompt'][i]}\n\n### Assistant: {example['completion'][i]}"
            output_texts.append(text)
        return output_texts

    def sft_train(self):
        if not os.path.isdir(SFT_FILE_PATH):
            print(f"Directory {SFT_FILE_PATH} is empty. Skipping SFT training.")
            if not os.listdir(SFT_FILE_PATH):
                print(
                    "Please populate the directory with data before running SFT training."
                )
            return
        dataset = load_dataset(SFT_FILE_PATH, data_files="sft_data.json", split="train")

        response_template_string = "\n### Assistant:"
        response_template_ids = self.tokenizer.encode(
            response_template_string, add_special_tokens=False
        )[2:]

        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=self.tokenizer
        )
        sft_config = SFTConfig(
            output_dir=TMP_SAVE_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_seq_length=8192,
        )

        if_load_adapter = check_load_adapter(self.llm_path)
        if if_load_adapter:
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(model, SAVE_DIR)
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            print("Loaded the adapter.")
        else:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.llm_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model = get_peft_model(model, peft_config)

        self.trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
            data_collator=collator,
            formatting_func=self.formatting_prompts_func,
        )

        print("Starting SFT training")
        self.trainer.train()

        self.trainer.save_model(TMP_SAVE_DIR)
        print(f"Training progress: 1.0")
        return model

    def ppo_train(self):
        if not os.path.isdir(PPO_FILE_PATH):
            print(f"Directory {PPO_FILE_PATH} is empty. Skipping PPO training.")
            if not os.listdir(PPO_FILE_PATH):
                print(
                    "Please populate the directory with data before running PPO training."
                )
            return
        dataset = load_dataset(PPO_FILE_PATH, data_files="ppo_data.json", split="train")

        def tokenize(sample):
            sample["query_ids"] = self.tokenizer.encode(sample["prompt"])
            sample["response_ids"] = self.tokenizer.encode(sample["completion"])
            sample["query"] = self.tokenizer.decode(sample["query_ids"])
            sample["response"] = self.tokenizer.decode(sample["response_ids"])
            sample["reward"] = float(sample["reward"])
            return sample

        dataset = dataset.map(tokenize, batch_size=False)
        dataset.set_format(
            type="torch", columns=["query_ids", "response_ids", "reward"]
        )

        if_load_adapter = check_load_adapter(self.llm_path)
        if if_load_adapter:
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(model, SAVE_DIR)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            print("Loaded the adapter.")
        else:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.llm_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                peft_config=peft_config,
            )

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        ref_model = create_reference_model(model)

        ppo_config = PPOConfig(
            ppo_epochs=1,
            whiten_rewards=True,
            batch_size=len(dataset),
            remove_unused_columns=False,
            mini_batch_size=1,
        )

        def collator(data):
            return {key: [d[key] for d in data] for key in data[0]}

        self.trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            data_collator=collator,
        )

        for _epoch, batch in tqdm(enumerate(self.trainer.dataloader)):
            print(f"The batch size is {len(batch['reward'])}...")
            query_tensors = batch["query_ids"]
            response_tensors = batch["response_ids"]
            rewards_tensors = [x.float() for x in batch["reward"]]

            stats = self.trainer.step(query_tensors, response_tensors, rewards_tensors)

        self.trainer.save_pretrained(TMP_SAVE_DIR)
        print("Training progress: 1.0")


def main(args):
    tuner = Tuner(args.tuning_mode, args.llm_path)
    tuner.train_func()


if __name__ == "__main__":
    args = parse_args()
    main(args)
