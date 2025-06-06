# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from nemo.collections.asr.parts.utils.manifest_utils import get_ctm_line

def min_max_token_wise_normalization(matrix):
    row_min = matrix.min(dim=1, keepdim=True).values
    row_max = matrix.max(dim=1, keepdim=True).values
    # Apply Min-Max normalization to each row
    normalized_matrix = (matrix - row_min) / (row_max - row_min)
    return normalized_matrix

def get_target_inds(dp_frame_inds, matrix, l_margin, r_margin, last_spk_frame_idx):
    left = max(0, dp_frame_inds[0] - l_margin + 1, last_spk_frame_idx)
    right = min(matrix.size(1) - 1, dp_frame_inds[-1] + r_margin)
    target_inds = torch.arange(left, right + 1)
    return target_inds

def get_attn_frames(idx, dp_frame_inds, matrix, l_margin, r_margin, last_spk_frame_idx, thres=0.5):
    # left = max(0, dp_frame_inds[0] - l_margin, last_spk_frame_idx+1)
    # right = min(matrix.size(1) - 1, dp_frame_inds[-1] + r_margin)
    # target_inds = torch.arange(left, right + 1)
    target_inds = get_target_inds(dp_frame_inds, matrix, l_margin, r_margin, last_spk_frame_idx)
    valid_mask = matrix[idx, target_inds] > thres
    if len(valid_mask) == 0 or torch.all(valid_mask).item() == False:
        frame_inds = dp_frame_inds
    else:
        frame_inds = target_inds[valid_mask]
    attn_vals = matrix[idx, frame_inds]
    return frame_inds, attn_vals

def get_word_seq_from_tokens_mono_spk(token_seq, attn_map, path_2d, rn_matrix, l_margin, r_margin, spk_wise_last_frame_inds, feat_frame_len_sec, spl_token_pattern=r'<\|spltoken\d+\|>'):
    sil_token="▁"
    last_spk_frame_idx = 0
    word_count = 0
    word_seq_dict = {}
    decimal=3
    attn_map = torch.zeros_like(rn_matrix)
    for idx, tok in enumerate(token_seq):
        dp_frame_inds = np.sort(path_2d[path_2d[:, 0] == idx, 1])
        frame_inds, attn_vals = get_attn_frames(idx, dp_frame_inds, rn_matrix, l_margin, r_margin, last_spk_frame_idx)
        attn_map[idx, frame_inds] = attn_vals

        if not (sil_token in tok and len(tok) == 1) and sil_token in tok:
            word_count += 1
            word_seq_dict[word_count] = {'word':[tok], 
                                         'start': round(feat_frame_len_sec * float(frame_inds[0]), decimal), 
                                         'end': round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)}
                                        #  'spk': spk}
            word_open = True
        elif sil_token not in tok and word_open:
            word_seq_dict[word_count]['word'].append(tok)
            word_seq_dict[word_count]['end'] = round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)
        else:
            word_open = False
        last_spk_frame_idx = idx + 1
    
    # Last item handler
    if word_open and word_seq_dict[word_count]['end'] is None:
        word_seq_dict[word_count]['end'] = round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)
        word_open = False
    return word_seq_dict, attn_map

def get_word_seq_from_tokens(token_seq, attn_map, path_2d, rn_matrix, l_margin, r_margin, spk_wise_last_frame_inds, feat_frame_len_sec, spl_token_pattern=r'<\|spltoken\d+\|>'):
    sil_token="▁"
    last_spk_frame_idx = 0
    word_count = 0
    word_seq_dict = {}
    decimal=3
    attn_map = torch.zeros_like(rn_matrix)
    for idx, tok in enumerate(token_seq):
        dp_frame_inds = np.sort(path_2d[path_2d[:, 0] == idx, 1])
        is_spl_token = contains_pattern = bool(re.search(spl_token_pattern, tok))
        if is_spl_token:
            spk = str(tok)
        else:
            spk = 'unknown'
        frame_inds, attn_vals = get_attn_frames(idx, dp_frame_inds, rn_matrix, l_margin, r_margin, last_spk_frame_idx=spk_wise_last_frame_inds[spk])
        # print(f"idx:{idx}, dp_frame_inds: {dp_frame_inds}, frame_inds: {frame_inds}, attn_vals: {attn_vals} spk: {spk}, spk_wise_last_frame_inds[spk] {spk_wise_last_frame_inds[spk]}, {frame_inds[-1]}, token:{tok}")
        attn_map[idx, frame_inds] = attn_vals

        if not (sil_token in tok and len(tok) == 1) and sil_token in tok:
            word_count += 1
            word_seq_dict[word_count] = {'word':[tok], 
                                         'start': round(feat_frame_len_sec * float(frame_inds[0]), decimal), 
                                         'end': round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal),
                                         'spk': spk}
            word_open = True
            spk_wise_last_frame_inds[spk] = frame_inds[-1].item()
        elif not is_spl_token and sil_token not in tok and word_open:
            word_seq_dict[word_count]['word'].append(tok)
            word_seq_dict[word_count]['end'] = round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)
            spk_wise_last_frame_inds[spk] = frame_inds[-1].item()
        else:
            word_open = False
    # Last item handler
    if word_open and word_seq_dict[word_count]['end'] is None:
        word_seq_dict[word_count]['end'] = round(feat_frame_len_sec * float(frame_inds[-1] + 1), decimal)
        word_open = False
    return word_seq_dict, attn_map

def generate_word_alignment_list_mono_spk(word_seq_dict, sil_token):
    word_alignment = []
    for word_count, word_info in word_seq_dict.items():
        word =''.join(word_info['word']).replace(sil_token, '')
        word_alignment.append([word,
                               word_info['start'], 
                               word_info['end']])
    return word_alignment

def generate_word_alignment_list(word_seq_dict, sil_token):
    word_alignment = []
    for word_count, word_info in word_seq_dict.items():
        word =''.join(word_info['word']).replace(sil_token, '')
        word_alignment.append([word,
                               word_info['start'], 
                               word_info['end'], 
                               word_info['spk']]
                              )
    return word_alignment

def get_word_alignment(
    matrix, 
    path_2d, 
    token_seq, 
    feat_frame_len_sec=0.08, 
    sil_token=None, 
    spl_token_pattern=r'<\|spltoken\d+\|>',
    decimal=2,
    l_margin: int = 0,
    r_margin: int = 0,
    max_spks: int = 10,
    is_multispeaker: bool = False,
    ):
    matrix = torch.tensor(matrix)
    attn_map = torch.zeros_like(matrix)
    word_count = -1
    spk = 'unknown'
    word_open = False
    rn_matrix = min_max_token_wise_normalization(matrix)
    spk_wise_last_frame_inds = { spl_token_pattern.replace('\\', '').replace('d+', f"{idx}"): -1 for idx in range(max_spks)}
    spk_wise_last_frame_inds[spk] = -1
    sil_token="▁"
    
    if is_multispeaker:
        word_seq_dict, attn_map = get_word_seq_from_tokens(token_seq=token_seq,
                                                        attn_map=attn_map,
                                                        path_2d=path_2d, 
                                                        rn_matrix=rn_matrix, 
                                                        l_margin=l_margin, 
                                                        r_margin=r_margin, 
                                                        spk_wise_last_frame_inds=spk_wise_last_frame_inds, 
                                                        feat_frame_len_sec=feat_frame_len_sec,
                                                        spl_token_pattern=spl_token_pattern)
        word_alignment = generate_word_alignment_list(word_seq_dict, sil_token)
    else:
        word_seq_dict, attn_map = get_word_seq_from_tokens_mono_spk(token_seq=token_seq,
                                                        attn_map=attn_map,
                                                        path_2d=path_2d, 
                                                        rn_matrix=rn_matrix, 
                                                        l_margin=l_margin, 
                                                        r_margin=r_margin, 
                                                        spk_wise_last_frame_inds=spk_wise_last_frame_inds, 
                                                        feat_frame_len_sec=feat_frame_len_sec,
                                                        spl_token_pattern=spl_token_pattern)
        word_alignment = generate_word_alignment_list_mono_spk(word_seq_dict, sil_token)
    return word_alignment, attn_map
        

def backtrack(direction, rows, cols):
    # Backtrack to find the path
    path = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if direction[i, j] == -1: # 'left':
            j -= 1
        else:
            i -= 1
    path.append((0, 0))  # add the starting point
    path.reverse()
    return path


def backtrack_diag(direction, rows, cols):
    # Backtrack to find the path
    path = []
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if direction[i, j] == -1:  # 'left'
            j -= 1
        elif direction[i, j] == 1:  # 'above'
            i -= 1
        # else:  # 'diagonal'
        elif direction[i, j] == 2:  # 'diagonal'
            i -= 1
            j -= 1
    path.append((0, 0))  # add the starting point
    path.reverse()
    return path

def dynamic_programming_max_sum_path(matrix):
        # Fill the DP table
    rows, cols = matrix.shape
    # Initialize DP table and direction table
    dp = torch.zeros_like(matrix, dtype=torch.float32)
    direction = torch.zeros((rows, cols), dtype=torch.int8)
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                dp[i, j] = matrix[i, j]
            elif i == 0:
                dp[i, j] = dp[i, j-1] + matrix[i, j]
                direction[i, j] = -1 # 'left'
            elif j == 0:
                dp[i, j] = dp[i-1, j] + matrix[i, j]
                direction[i, j] = 1 # 'above'
            else:
                if dp[i-1, j] > dp[i, j-1]:
                    dp[i, j] = dp[i-1, j] + matrix[i, j]
                    direction[i, j] =  1 # 'above'
                else:
                    dp[i, j] = dp[i, j-1] + matrix[i, j]
                    direction[i, j] = -1 # 'left'
    return direction, rows, cols

def dynamic_programming_max_sum_path_diag(matrix):
    # Fill the DP table
    rows, cols = matrix.shape
    # Initialize DP table and direction table
    dp = torch.zeros_like(matrix, dtype=torch.float32)
    direction = torch.zeros((rows, cols), dtype=torch.int8)

    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                dp[i, j] = matrix[i, j]
            elif i == 0:
                dp[i, j] = dp[i, j-1] + matrix[i, j]
                direction[i, j] = -1 # 'left'
            elif j == 0:
                dp[i, j] = dp[i-1, j] + matrix[i, j]
                direction[i, j] = 1 # 'above'
            else:
                # Check left, above, and diagonal (top-left)
                left = dp[i, j-1]
                above = dp[i-1, j]
                diagonal = dp[i-1, j-1]

                if above >= left and above >= diagonal:
                    print(f"above")
                    dp[i, j] = above + matrix[i, j]
                    direction[i, j] = 1 # 'above'
                elif left >= above and left >= diagonal:
                    print(f"left")
                    dp[i, j] = left + matrix[i, j]
                    direction[i, j] = -1 # 'left'
                else:
                    print(f"diag")
                    dp[i, j] = diagonal + matrix[i, j]
                    direction[i, j] = 2 # 'diagonal'

    return direction, rows, cols


def run_dp_for_alignment(matrix):
    direction, rows, cols = dynamic_programming_max_sum_path(matrix)
    path = backtrack(direction, rows, cols)
    path_2d = np.array(path)
    return path_2d

def save_plot_imgs(path_2d, layer_info, matrix, filename, tokens, FS=1.5):
    # Set the y-axis ticks and labels based on the tokens
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
        
    plt.plot(path_2d[:, 1], path_2d[:, 0], marker='o', color='black', linewidth=1.0, alpha=0.25, markersize=1)
    plt.imshow(matrix, cmap='jet', interpolation='none')
    plt.text(x=100, y=50, s=f'Layer {layer_info}', color='white', fontsize=12, 
         bbox=dict(facecolor='red', alpha=0.5))  # Adjust x, y, and text properties
    y_ticks = np.arange(len(tokens))  # Assuming `tokens` is the list of strings for each y-tick
    plt.yticks(y_ticks, tokens, fontsize=FS)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=600)
    print(f"Saved the image to | {filename}")
    plt.clf() 
    plt.close()

def save_attn_imgs(
    cfg,
    asr_model, 
    transcriptions, 
    num_layers: int, 
    tensor_attn_dict, 
    tokens, 
    idx: int = 0,
    pre_tokens_len=5,
    end_cut = 2,
    layer_stt = 8,
    layer_end= 16,
    is_multispeaker=True,
    ): 
    """
    https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynb
    """
    if cfg.get("diar_pred_model_path", None) is None:
        is_multispeaker = False
    # Skip the end token because end token does not generate an attention step.
    token_seq = asr_model.tokenizer.ids_to_tokens(transcriptions[0].y_sequence.cpu().numpy())[pre_tokens_len:-1]
    token_seq = np.array(token_seq)
    
    token_ids = transcriptions[0].y_sequence[pre_tokens_len:]
    (spl_stt, spl_end) = (13, 30)
    spk_token_inds = torch.nonzero((token_ids >= spl_stt) & (token_ids <= spl_end)).squeeze()
    
    # if attn_block is not None:
    avg_list = []
    for k in range(layer_stt, layer_end+1):
        attn_block = tensor_attn_dict[k]
        if layer_stt <= k <= layer_end:
            avg_list.append( torch.mean( attn_block[idx, :, 0, :, :], dim=0).unsqueeze(0) )
        
        output_dir = "/home/taejinp/Downloads/multispeaker_canary_imshow/"
        _img = attn_block[idx, :, 0, :, :]
        img = torch.mean(_img, dim=0).cpu()
        non_spk_mask = torch.ones(img.size(0), dtype=bool)
        all_mask = torch.ones(img.size(0), dtype=bool)
        non_spk_mask[spk_token_inds] = False
        
        # Convert tensor to numpy array
        img = img.numpy()
        if False:
        # if True:
            # for (token_type, mask) in [('Words', non_spk_mask), ('Spks', spk_token_inds), ('All', all_mask)]:
            for (token_type, mask) in [('All', all_mask)]:
                if not is_multispeaker and token_type == 'Spks':
                    continue
                filename = os.path.join(output_dir, f'image_layer-{k+1}th-{token_type}_batch-{idx}.png')
                matrix = torch.tensor(img[mask, :-end_cut])
                path_2d = run_dp_for_alignment(matrix)
                # save_plot_imgs(path_2d=path_2d, layer_info=f"layer-{k+1}th", matrix=matrix, filename=filename, tokens=token_seq[mask])

    # for (token_type, mask) in [('Words', non_spk_mask), ('Spks', spk_token_inds), ('All', all_mask)]:
    for (token_type, mask) in [('All', all_mask)]:
        if not is_multispeaker and token_type == 'Spks':
            continue
        layer_avg_attn_block = torch.mean(torch.vstack(avg_list), dim=0).cpu().numpy()
        if False:
            matrix = torch.tensor(layer_avg_attn_block[mask, :-end_cut])
            filename = os.path.join(output_dir, f'image_layerAVG-{token_type}-stt{layer_stt}_end{layer_end}.png')
            path_2d = run_dp_for_alignment(matrix)
            save_plot_imgs(path_2d=path_2d, layer_info=f"layer-avg-stt{layer_stt}_end{layer_end}", matrix=matrix, filename=filename, tokens=token_seq[mask])
    matrix_all = layer_avg_attn_block[:, :-end_cut]
    matrix = torch.tensor(layer_avg_attn_block[mask, :-end_cut])
    path_2d_all = run_dp_for_alignment(matrix)
    sil_token = asr_model.tokenizer.ids_to_tokens([31])[0]
    word_alignment, attn_map = get_word_alignment(matrix=matrix_all, path_2d=path_2d_all, token_seq=token_seq, is_multispeaker=cfg.is_multispeaker, sil_token=sil_token)
    if False:
        thr_filename = os.path.join(output_dir, f'image_THRattn-{token_type}-stt{layer_stt}_end{layer_end}.png')
        save_plot_imgs(path_2d=path_2d, layer_info=f"layer-avg-stt{layer_stt}_end{layer_end}", matrix=attn_map, filename=thr_filename, tokens=token_seq[mask])
    return word_alignment

def write_ctm(filepaths, ctm_filepath, cfg, word_alignments, decimal=2, str_pattern=r'<\|spltoken\d+\|>'):
    str_pattern = str_pattern.replace("\\", '')
    left_str, right_str = str_pattern.split('d+')[0], str_pattern.split('d+')[1]
    
    filepath_list = open(filepaths).readlines()
    ctm_output_list, rttm_output_list = [], []
    for idx, json_string in enumerate(filepath_list):
        meta_dict = json.loads(json_string)
        uniq_id = os.path.basename(meta_dict['audio_filepath']).split('.')[0]
        # output_folder = os.path.dirname(cfg.output_filename)
        output_folder = os.path.dirname(ctm_filepath)
        ctm_filename = os.path.join(output_folder, f"{uniq_id}.ctm")
        rttm_filename = os.path.join(output_folder, f"{uniq_id}.rttm")
        for word_line in word_alignments:
            word = word_line[0]
            dur = round(word_line[2] - word_line[1], decimal) - 0.01
            start = round(word_line[1], decimal)
            if cfg.is_multispeaker:
                spk_str = word_line[3]
                if 'unknown' in spk_str or spk_str is None:
                    spk_token_int = 0
                else: 
                    spk_token_int = int(spk_str.replace(left_str,'').replace(right_str, ''))
            else:
                spk_str = 'NA'
                spk_token_int = 0  # for silence token in CTM file
            rttm_line = f"SPEAKER {uniq_id} {spk_token_int} {start:.3f} {dur:.3f} <NA> <NA> {spk_str} <NA> <NA> {word}\n"
            ctm_line = get_ctm_line(
                    source=uniq_id + f"_spk{spk_token_int}",
                    channel=f"{spk_token_int}",
                    start_time=start,
                    duration=dur,
                    token=word,
                    conf=1.0,
                    type_of_token="lex",
                    speaker=f"{spk_str}",
            )
            ctm_output_list.append(ctm_line)
            rttm_output_list.append(rttm_line)
            
    with open(ctm_filename, 'w') as f:
        f.write(''.join(ctm_output_list))
        print(f"Saved CTM to | {ctm_filename}")
    with open(rttm_filename, 'w') as f:
        f.write(''.join(rttm_output_list))