#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import platform
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../../'))

from project_settings import project_path

hf_hub_cache = (project_path / "cache/huggingface/hub").as_posix()

os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
import numpy as np
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch.utils.data.dataloader import DataLoader

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-uncased",
        choices=["bert-base-uncased"],
        type=str
    )
    parser.add_argument(
        "--dataset_path",
        default="banking77",
        choices=["banking77"],
        type=str
    )
    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--output_file", default="normalize.txt", type=str)

    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--warm_up_vector_count", default=2, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    dataset_dict = load_dataset(
        path=args.dataset_path,
        cache_dir=args.dataset_cache_dir,
    )
    print(dataset_dict)

    # model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModel.from_pretrained(args.pretrained_model_name_or_path)
    model.eval()

    # map
    def encode_with_truncation(examples):
        outputs = tokenizer.__call__(examples["text"],
                                     truncation=True,
                                     # padding="max_length",
                                     max_length=args.max_seq_length,
                                     return_special_tokens_mask=True,
                                     )
        return outputs

    dataset_dict = dataset_dict.map(
        encode_with_truncation,
        batched=True,
        drop_last_batch=True,
        batch_size=1000,
        keep_in_memory=False,
        num_proc=None if platform.system() == 'Windows' else os.cpu_count() // 2,
    )
    dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

    count = 0
    warm_up_vectors = list()
    for sample in dataset_dict["train"]:
        count += 1

        input_ids: torch.Tensor = sample["input_ids"]
        attention_mask: torch.Tensor = sample["attention_mask"]
        input_ids = input_ids.unsqueeze(dim=0)
        attention_mask = attention_mask.unsqueeze(dim=0)

        with torch.no_grad():
            output: BaseModelOutputWithPoolingAndCrossAttentions = model.forward(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
        pooler_output = output.pooler_output.numpy().squeeze(0)
        warm_up_vectors.append(pooler_output)

        if count == args.warm_up_vector_count:
            break

    # initial covariance matrix.
    m = np.array(warm_up_vectors, dtype=np.float32)
    mean = np.mean(m, axis=0)
    m = m - mean
    cov = np.dot(np.transpose(m), m)
    print(cov.shape)

    # covariance matrix incremental update.
    last_mean = mean
    last_cov = cov
    count = 0
    for sample in tqdm(dataset_dict["train"]):
        count += 1
        if count <= args.warm_up_vector_count:
            continue

        input_ids: torch.Tensor = sample["input_ids"]
        attention_mask: torch.Tensor = sample["attention_mask"]
        input_ids = input_ids.unsqueeze(dim=0)
        attention_mask = attention_mask.unsqueeze(dim=0)

        with torch.no_grad():
            output: BaseModelOutputWithPoolingAndCrossAttentions = model.forward(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
        pooler_output = output.pooler_output.detach().numpy().squeeze(0)

        x = pooler_output
        x_minus = np.expand_dims(x - last_mean, axis=0)

        last_cov += (x_minus.T * x_minus) * (count - 1) / count
        last_mean = ((count - 1) * last_mean + x) / count

        count += 1

    print(last_cov.shape)
    print(last_mean.shape)

    # divide by (n-1).
    cov = last_cov / (count - 1)
    mean = last_mean

    # singular value decomposition
    u, s, vh = np.linalg.svd(cov)
    w = np.dot(u, np.diag(1 / np.sqrt(s)))
    w = w[:, :64]

    kernel = w
    bias = - mean
    print(kernel.shape)
    print(bias.shape)
    return


if __name__ == '__main__':
    main()
