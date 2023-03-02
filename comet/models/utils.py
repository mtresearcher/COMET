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
from typing import List

import torch
from torch.utils.data import Sampler
from transformers.utils import ModelOutput
from typing import List, Optional


def add_context(orig_txt: List[str],
                context: List[str],
                doc_ids: List[str],
                sep_token: str = "</s>",
                ws: int = 2,
                use_future_context: Optional[bool] = False) -> List[str]:
    """Function that adds the previous sentences as context to the current sentence, respecting document boundaries
    :param orig_txt: the original text
    :param context: the text from which the context will be taken (same as orig_txt for source/reference)
    :param doc_ids: the document where each segment belongs to
    :param sep_token: the separator token of the tokenizer for the specific model
    :param ws: the window size, maximum of the previous sentences to be considered as context
    :param use_future_context: uses future context together with previous context
    :return: the original text augmented with context
    """
    if not (len(orig_txt) == len(context) == len(doc_ids)):
        raise Exception(f'Lengths should match: '
                        f'len(orig_txt)={len(orig_txt)}, '
                        f'len(context)={len(context)}, '
                        f'len(doc_ids)={len(doc_ids)}')
    i, k = 0, 0
    augm_txt = []
    doc_id = doc_ids[0]
    while i < len(orig_txt):
        if doc_ids[i] == doc_id:
            if use_future_context:
                context_window = context[i - int(min(k, ws)/2) : i + int(min(k, ws)/2)]
            else:
                context_window = context[i - min(k, ws): i]
            augm_txt.append(" {} ".format(sep_token).join(context_window + [orig_txt[i]]))
            i += 1
        else:
            doc_id = doc_ids[i]
            k = -1
        k += 1
    return augm_txt


class Prediction(ModelOutput):
    """Renamed ModelOutput"""

    pass


class Target(ModelOutput):
    """Renamed ModelOutput into Targets to keep same behaviour"""

    pass


class OrderedSampler(Sampler[int]):
    """Sampler that returns the indices in a deterministic order."""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def flatten_metadata(metadata):
    """Metadata from the model output can be in various forms and this function
    will gather all metadata and flatten everything.
    """
    metadata = Prediction(**{k: [dic[k] for dic in metadata] for k in metadata[0]})
    for k, v in metadata.items():
        if torch.is_tensor(v[0]):
            # If we have tensors we can use cat to flatten them.
            metadata[k] = torch.cat(v, dim=0).tolist()
        else:
            # for other predictions such as word tags we have to flatten the list.
            metadata[k] = [item for sublist in v for item in sublist]
    return metadata


def restore_list_order(sorted_list, sort_ids):
    """Restores the original ids of a given list."""
    unsorted_list = [None for _ in range(len(sorted_list))]
    for i, s in zip(sort_ids, sorted_list):
        unsorted_list[i] = s
    return unsorted_list
