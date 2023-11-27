#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with discriminator:
"""

import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from pplm_classification_head import ClassificationHead

SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    "quiet": QUIET,
    "regular": REGULAR,
    "verbose": VERBOSE,
    "very_verbose": VERY_VERBOSE,
}


DISCRIMINATOR_MODELS_PARAMS = {
    "toxicity": {
        "path": "/home/repos/PPLM/paper_code/discrim_models/toxicity_classifierhead.pt",
        "pretrained_model": "gpt2-medium",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"toxic": 1, "non_toxic": 0},
        "default_class": 0,
    },
}


def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(
                logits < batch_mins, torch.ones_like(logits) * 0.0, logits
            )
        return torch.where(
            logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits
        )


def perturb_past(
    past,
    model,
    last,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    classifier=None,
    class_label=None,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
    verbosity_level=REGULAR,
):
    # Generate inital perturbed past
    grad_accumulator = [(np.zeros(p[0].shape).astype("float32")) for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.0, 1.0 + SMALL_CONST, 1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    # (batch_size, num_heads, seq_length, head_dim)
    _, _, curr_length, _ = past[0][0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
            tuple(past[0][0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0][0].shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past[0][0].shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past[0][0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0][0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        # len: 24(?)
        # each element: [batch, num_heads, seq, d_head]
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past_key = list(
            map(add, [p[0] for p in past], curr_perturbation)
        )
        perturbed_past_value = list(
            map(add, [p[1] for p in past], curr_perturbation)
        )
        perturbed_past = list(zip(perturbed_past_key, perturbed_past_value))
        _, _, curr_length, _ = curr_perturbation[0].shape
        _output = model(last, past_key_values=perturbed_past)
        all_logits = _output["logits"]
        all_hidden = _output["hidden_states"]
        hidden = all_hidden[-1]
        new_accumulated_hidden = (
            accumulated_hidden + torch.sum(hidden, dim=1).detach()
        )
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        ce_loss = torch.nn.CrossEntropyLoss()
        # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
        curr_unpert_past = unpert_past

        # [batch, seq, vocab]
        curr_probs = torch.unsqueeze(probs, dim=1)

        # [vocab, d_model]
        wte = model.resize_token_embeddings()
        for _ in range(horizon_length):
            # [batch, seq, d_model]
            inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
            _output = model(
                past_key_values=curr_unpert_past,
                inputs_embeds=inputs_embeds,
            )
            curr_unpert_past = _output["past_key_values"]
            curr_all_hidden = _output["hidden_states"]
            curr_hidden = curr_all_hidden[-1]
            new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                curr_hidden, dim=1
            )

        # classifier: 1024 --> 2
        # new_accumulated_hidden: [batch, d_model]
        # prediction: [batch, 2]
        prediction = classifier(
            new_accumulated_hidden / (curr_length + 1 + horizon_length)
        )

        label = torch.tensor(
            prediction.shape[0] * [class_label],
            device=device,
            dtype=torch.long,
        )
        discrim_loss = ce_loss(prediction, label)
        if verbosity_level >= VERY_VERBOSE:
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
        loss += discrim_loss
        loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                unpert_probs
                + SMALL_CONST
                * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = (
                SMALL_CONST
                * (probs <= SMALL_CONST).float().to(device).detach()
            )
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (
                    corrected_probs * (corrected_probs / unpert_probs).log()
                ).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(" kl_loss", kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(" pplm_loss", (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        grad_norms = [
            (torch.norm(p_.grad * window_mask) + SMALL_CONST)
            for index, p_ in enumerate(curr_perturbation)
        ]

        # normalize gradients
        grad = [
            -stepsize
            * (p_.grad * window_mask / grad_norms[index] ** gamma)
            .data.cpu()
            .numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append((p_[0].detach(), p_[1].detach()))
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past_key = list(map(add, [p[0] for p in past], grad_accumulator))
    pert_past_value = list(map(add, [p[1] for p in past], grad_accumulator))
    pert_past = list(zip(pert_past_key, pert_past_value))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
    name: Optional[str],
    class_label: Union[str, int],
    device: str,
    verbosity_level: int = REGULAR,
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params["class_size"], embed_size=params["embed_size"]
    ).to(device)
    if "url" in params:
        breakpoint()
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError(
            "Either url or path have to be specified "
            "in the discriminator model parameters"
        )
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device)
    )
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def full_text_generation(
    model,
    tokenizer,
    context,
    config,
    verbosity_level=REGULAR,
):
    num_samples = config.num_samples
    device = config.device
    discrim = config.discrim
    class_label = config.class_label

    classifier, class_id = get_classifier(discrim, class_label, device)
    assert classifier is not None

    unpert_gen_tok_text, _, _ = generate(
        model,
        tokenizer,
        config,
        context=context,
        verbosity_level=verbosity_level,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate(
            model,
            tokenizer,
            config,
            context=context,
            classifier=classifier,
            class_label=class_id,
            verbosity_level=verbosity_level,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()

    return (
        unpert_gen_tok_text,
        pert_gen_tok_texts,
        discrim_losses,
        losses_in_time,
    )


def generate(
    model,
    tokenizer,
    config,
    classifier=None,
    class_label=None,
    context=None,
    past=None,
    verbosity_level=REGULAR,
):
    device = config.device
    length = config.length
    grad_length = config.grad_length
    stepsize = config.stepsize
    num_iterations = config.num_iterations
    temperature = config.temperature
    sample = config.sample
    gm_scale = config.gm_scale
    top_k = config.top_k
    perturb = classifier is not None

    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    for i in range(length):

        # if past is not None and isinstance(past[0], tuple):
        #    breakpoint()

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _output = model(output_so_far[:, :-1])
                past = _output["past_key_values"]

        _output = model(output_so_far)
        unpert_logits = _output["logits"]
        unpert_past = _output["past_key_values"]
        unpert_all_hidden = _output["hidden_states"]
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are above grad max length
        current_stepsize = stepsize
        if i >= grad_length:
            current_stepsize = stepsize * 0

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    classifier=classifier,
                    class_label=class_label,
                    num_iterations=num_iterations,
                    horizon_length=config.horizon_length,
                    window_length=config.window_length,
                    decay=config.decay,
                    gamma=config.gamma,
                    kl_scale=config.kl_scale,
                    device=device,
                    verbosity_level=verbosity_level,
                )
                loss_in_time.append(loss_this_iter)
                #breakpoint()
            else:
                pert_past = past

        _output = model(last, past_key_values=pert_past)
        pert_logits = _output["logits"]
        past = _output["past_key_values"]
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor(
                [class_label], device=device, dtype=torch.long
            )
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy(),
                )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * (
                unpert_probs ** (1 - gm_scale)
            )  # + SMALL_CONST
            pert_probs = top_k_filter(
                pert_probs, k=top_k, probs=True
            )  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last
            if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )

    #if verbosity_level >= REGULAR:
    #    print(tokenizer.decode(output_so_far.tolist()[0]))
    return output_so_far, unpert_discrim_loss, loss_in_time


def run_pplm_example(
    cond_text,
    config,
    verbosity="regular",
):
    # set Random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert config.discrim is not None
    assert (
        config.pretrained_model
        == DISCRIMINATOR_MODELS_PARAMS[config.discrim]["pretrained_model"]
    )

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        config.pretrained_model, output_hidden_states=True
    )
    model.to(config.device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.pretrained_model)
    #tokenizer.padding_size = "left"
    #tokenizer.pad_token_id = tokenizer.eos_token_id

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    tokenized_cond_text = tokenizer.encode(
        tokenizer.bos_token + cond_text, add_special_tokens=False
    )

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model,
        tokenizer,
        tokenized_cond_text,
        config,
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    if verbosity_level >= REGULAR:
        print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    generated_texts = []

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        # untokenize unperturbed text
        pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

        print("= Perturbed generated text {} =".format(i + 1))
        print(pert_gen_text)
        print()

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return


if __name__ == "__main__":
    config = {
        "pretrained_model": "gpt2-medium",
        "num_samples": 1,
        "discrim": "toxicity",
        "class_label": 1,
        "length": 32,
        "stepsize": 0.2,
        "temperature": 1,
        "top_k": 10,
        "sample": False,
        "num_iterations": 10,
        "grad_length": 10000,
        "window_length": 0,
        "horizon_length": 1,
        "decay": False,
        "gamma": 1.0,
        "gm_scale": 0.95,
        "kl_scale": 0.1,
        "seed": 0,
    }

    class Config:
        def __init__(self, **attrs):
            self.__dict__.update(attrs)

    run_pplm_example("Yesterday", Config(**config))
    run_pplm_example("Whether", Config(**config))
    run_pplm_example("According to", Config(**config))
    run_pplm_example("In all seriousness,", Config(**config))
    run_pplm_example("My goals in life", Config(**config))
