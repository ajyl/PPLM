"""
Module Doc String
"""

import sys
import os
from tqdm import tqdm
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gen_toxic_data.run_batch import run_pplm_batch, get_classifier
from constants import DATA_DIR


def serialize(tensor):
    return tensor.detach().cpu().numpy().tolist()


def _generate_pairwise_data(config):
    """ Generate data """
    output_dir = config.output_dir
    splice_dir = config.splice_dir
    data_type = config.data_type
    task_number = config.task_number
    batch_size = config.batch_size
    max_prompt_size = config.max_prompt_size
    max_generation_size = config.max_generation_size

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    with open(
        os.path.join(splice_dir, f"{data_type}_{task_number}.jsonl"), "r"
    ) as file_p:
        data = file_p.readlines()

    data = [x.strip() for x in data]
    num_data = len(data)
    if config.num_data is not None:
        num_data = config.num_data

    print("Loading GPT2...")
    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        config.pretrained_model, output_hidden_states=True
    )
    model.to(config.device)
    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    print("Loading tokenizer...")
    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.pretrained_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading classifier...")
    classifier, class_id = get_classifier(
        config.discrim, config.class_label, config.device
    )
    pairwise_data = []
    for idx in tqdm(range(0, num_data, batch_size)):
        batch = [json.loads(x) for x in data[idx : idx + batch_size]]
        prompts = [x[0] for x in batch]
        gold = [x[1] for x in batch]

        # figure out conditioning text
        tokenized_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_size,
        )
        _tokenized_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_size + max_generation_size,
        )
        tokenized_gold = _tokenized_prompts["input_ids"][:, max_prompt_size:]

        unpert_gen_toks, pert_gen_toks = run_pplm_batch(
            model, tokenizer, tokenized_prompts, classifier, class_id, config
        )
        pert_gen_toks = pert_gen_toks[0]

        _prompt = tokenizer.batch_decode(
            tokenized_prompts["input_ids"], skip_special_tokens=True
        )
        _gold = tokenizer.batch_decode(
            tokenized_gold, skip_special_tokens=True
        )
        # untokenize unperturbed text
        unpert_gen_text = tokenizer.batch_decode(
            unpert_gen_toks, skip_special_tokens=True
        )

        # untokenize unperturbed text
        pert_gen_text = tokenizer.batch_decode(
            pert_gen_toks, skip_special_tokens=True
        )

        print("=============")
        for _idx, gen in enumerate(unpert_gen_text):
            print("Prompt:", _prompt[_idx])
            print("Gold:", _gold[_idx])
            print("GPT2:", gen)
            print("PPLM:", pert_gen_text[_idx])
            print()

        assert unpert_gen_toks.shape[0] == pert_gen_toks.shape[0]
        assert unpert_gen_toks.shape[0] == len(_prompt)
        assert unpert_gen_toks.shape[0] == len(_gold)
        assert unpert_gen_toks.shape[0] == len(unpert_gen_text)
        assert unpert_gen_toks.shape[0] == len(pert_gen_text)

        for sample_idx in range(unpert_gen_toks.shape[0]):
            pairwise_data.append(
                {
                    "prompt_text": prompts[sample_idx],
                    "prompt_input_ids": serialize(
                        tokenized_prompts["input_ids"][sample_idx]
                    ),
                    "gold_text": gold[sample_idx],
                    "gold_input_ids": serialize(tokenized_gold[sample_idx]),
                    "unpert_gen_text": unpert_gen_text[sample_idx],
                    "unpert_gen_token_ids": serialize(unpert_gen_toks),
                    "pert_gen_text": pert_gen_text[sample_idx],
                    "pert_gen_toks": serialize(pert_gen_toks[sample_idx]),
                }
            )

    output_filepath = os.path.join(
        output_dir, f"{data_type}_{task_number}.jsonl"
    )
    with open(output_filepath, "w") as file_p:
        for sample in pairwise_data:
            file_p.write(json.dumps(sample))
            file_p.write("\n")


def generate_pairwise_data(task_number):
    """
    Generate pairwise data.
    """
    config = {
        # PPLM Configs
        "pretrained_model": "gpt2-medium",
        "num_samples": 1,
        "discrim": "toxicity",
        "class_label": 1,
        "stepsize": 0.4,
        "temperature": 1,
        "top_k": 10,
        "sample": True,
        "num_iterations": 50,
        "grad_length": 100000,
        "window_length": 0,
        "horizon_length": 1,
        "decay": False,
        "gamma": 1.0,
        "gm_scale": 0.95,
        "kl_scale": 0.1,
        "seed": 0,
        "verbosity": "regular",
        # Generation script configs
        "data_type": "train",
        "task_number": task_number,
        "splice_dir": os.path.join(
            DATA_DIR, "repetitions/wiki103_splices_w_next_sents"
        ),
        "output_dir": os.path.join(
            DATA_DIR, "toxicity/wiki103_toxicity_pairwise"
        ),
        "batch_size": 32,
        "max_prompt_size": 5,
        "max_generation_size": 32,
        "num_data": 32,
    }

    class Config:
        def __init__(self, **attrs):
            self.__dict__.update(attrs)

    config = Config(**config)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.length = config.max_generation_size
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    _generate_pairwise_data(config)


def main():
    """ Driver """

    task_number = sys.argv[1]
    generate_pairwise_data(task_number)


if __name__ == "__main__":
    main()
