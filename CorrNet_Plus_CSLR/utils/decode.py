import os
import time
import torch
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder


class Decode(object):

    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id

        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        vocab_list = list(vocab)
        vocab_list[blank_id] = ""

        self.ctc_decoder = build_ctcdecoder(
            vocab_list,
            kenlm_model_path=None,
            alpha=0,
            beta=0,
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            seq_len = vid_lgt[batch_idx].item()
            logits_np = nn_output[batch_idx][:seq_len].cpu().numpy()
            top_result_str = self.ctc_decoder.decode(logits_np, beam_width=10)
            char_indices = [ord(c) - 20000 for c in top_result_str]
            if len(char_indices) > 0:
                collapsed = [x[0] for x in groupby(char_indices)]
                filtered = [x for x in collapsed if x != self.blank_id]
            else:
                filtered = []
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(filtered)
                if int(gloss_id) in self.i2g_dict
            ])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(max_result)
            ])
        return ret_list