# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
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
import torch
from typing import List, Optional, Union


def find_start_inds(
        mask: torch.Tensor,
        tokens: torch.Tensor,
        separator_index: int,
) -> Union[List[int], torch.Tensor]:
    """Function that returns a list containing the start indices of each sentence
       for multi-sentence sequences and a new mask to omit all context sentences
       from the pooling function.
    :param mask: Padding mask [batch_size x seq_length]
    :param tokens: Word ids [batch_size x seq_length]
    :param separator_index: Separator token index.
    """
    start_inds = []
    # everything is set to 1 in mask
    ctx_mask = mask
    for i, sent in enumerate(tokens):
        # find all separator tokens in the sequence
        separators = (sent == separator_index).nonzero()
        if len(separators) > 1:
            # if there are more than one find where the last sentence starts
            ind = separators[-2].cpu().numpy().item()
            start_inds.append(ind)
            # set the mask values to 0 for the current sentence
            ctx_mask[i, 1:ind+1] = 0
        else:
            start_inds.append(0)
    return start_inds, ctx_mask


def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
    separator_index: Optional[int],
    doc_eval: Optional[bool] = False
) -> torch.Tensor:
    """Average pooling method.

    Args:
        :param tokens: (torch.Tensor): Word ids [batch_size x seq_length]
        :param embeddings: (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        :param mask: (torch.Tensor) Padding mask [batch_size x seq_length]
        :param padding_index: (torch.Tensor): Padding value
        :param separator_index: Separator token index
        :param doc_eval: Document-level evaluation
    Return:
        torch.Tensor: Sentence embedding
    """
    ctx_mask = mask
    if doc_eval:
        start_indices, ctx_mask = find_start_inds(mask, tokens, separator_index)
        wordemb = mask_fill(0.0, tokens, embeddings, padding_index, start_indices)
    else:
        wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = ctx_mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


def max_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
    separator_index: Optional[int],
    doc_eval: Optional[bool] = False
) -> torch.Tensor:
    """Max pooling method.

    Args:
        :param tokens: (torch.Tensor): Word ids [batch_size x seq_length]
        :param embeddings: (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        :param mask: (torch.Tensor) Attention mask [batch_size x seq_length]
        :param padding_index: (torch.Tensor): Padding value
        :param separator_index: Separator token index
        :param doc_eval: Document-level evaluation

    Return:
        torch.Tensor: Sentence embedding
    """
    if doc_eval:
        start_indices, _ = find_start_inds(mask, tokens, separator_index)
        wordemb = mask_fill(float("-inf"), tokens, embeddings, mask, padding_index, start_indices).max(dim=1)[0]
    else:
        wordemb = mask_fill(float("-inf"), tokens, embeddings, mask, padding_index).max(dim=1)[0]

    return wordemb


def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
    start_indices: Optional[list] = None,
) -> torch.Tensor:
    """Method that masks embeddings representing padded elements.

    Args:
        :param fill_value: (float) the value to fill the embeddings belonging to padded tokens
        :param tokens: (torch.Tensor) Word ids [batch_size x seq_length]
        :param embeddings: (torch.Tensor) Word embeddings [batch_size x seq_length x
            hidden_size]
        :param padding_index: (int)Padding value.
        :param start_indices: Start of sentence indices.

    Return:
        torch.Tensor: Word embeddings [batch_size x seq_length x hidden_size]
    """
    if start_indices is not None:
        padding_mask = tokens.eq(padding_index).unsqueeze(-1)
        padding_mask_ctx = torch.zeros(tokens.shape, dtype=torch.bool, device=padding_mask.device)
        for i, start in enumerate(start_indices):
            padding_mask_ctx[i, 1: start + 1] = True
        padding_mask = torch.logical_or(padding_mask, padding_mask_ctx.unsqueeze(-1))
    else:
        padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
