# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
from dataclasses import dataclass, is_dataclass, field
from typing import Optional, Union, List, Tuple, Dict, Any

import torch
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import open_dict
from lhotse.dataset.collation import collate_matrices


import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

from copy import deepcopy
from nemo.collections.asr.parts.utils.diarization_utils import read_seglst, OnlineEvaluation
from nemo.utils import logging

from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.core.config import hydra_runner

from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR, get_multi_talker_samples_from_manifest
from nemo.collections.asr.parts.utils.speaker_utils import (
audio_rttm_map as get_audio_rttm_map,
rttm_to_labels,
)
from nemo.collections.asr.parts.utils.diarization_utils import (
print_sentences,
get_color_palette,
write_txt,
)
from nemo.collections.asr.data.audio_to_diar_label import get_frame_targets_from_rttm, extract_frame_info_from_rttm


from typing import List, Optional
from dataclasses import dataclass
from collections import OrderedDict
import itertools

import time
from functools import wraps
import math

@dataclass
class DiarizationConfig:
    # Required configs
    diar_model_path: Optional[str] = None  # Path to a .nemo file
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    max_num_of_spks: Optional[int] = 4
    parallel_speaker_strategy: bool = False
    
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    eval_mode: bool = False
    no_der: bool = False
    out_rttm_dir: Optional[str] = None
    opt_style: Optional[str] = None
    
    # General configs
    session_len_sec: float = -1 # End-to-end diarization session length in seconds
    num_workers: int = 8
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True # If True, postprocessing will be bypassed
    log: bool = True # If True, log will be printed
    
    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25 # Collar in seconds for DER calculation
    ignore_overlap: bool = False # If True, DER will be calculated only for non-overlapping segments
    
    # Streaming diarization configs
    streaming_mode: bool = True # If True, streaming diarization will be used.
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ASR Configs
    asr_model: Optional[str] = None
    diar_model: Optional[str] = None
    device: str = 'cuda'
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = False
    compare_vs_offline: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None # ["ctc", "rnnt"]
    att_context_size: Optional[list] = None
    generate_realtime_scripts: bool = True
    
    word_window: int = 50
    fix_speaker_assignments: bool = False
    sentence_break_threshold_in_sec: float = 10000.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: str = "./"

    ignored_initial_frame_steps: int = 5
    verbose: bool = False

    feat_len_sec: float = 0.01
    finetune_realtime_ratio: float = 0.01

    spk_supervision: str = "diar" # ["diar", "rttm"]
    binary_diar_preds: bool = False


def format_time(seconds):
    minutes = math.floor(seconds / 60)
    sec = seconds % 60
    return f"{minutes}:{sec:05.2f}"

def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

def add_delay_for_real_time(cfg, chunk_audio, session_start_time, feat_frame_count, loop_end_time, loop_start_time):
    """ 
    Add artificial delay for real-time mode by calculating the time difference between 
    the current time and the session start time..

    Args:
        cfg (DiarizationConfig): The configuration object. 
    """
    time_diff = max(0, (time.time() - session_start_time) - feat_frame_count * cfg.feat_len_sec)
    eta_min_sec = format_time(time.time() - session_start_time)
    logging.info(f"[   REAL TIME MODE   ] min:sec - {eta_min_sec} "
                    f"Time difference for real-time mode: {time_diff:.4f} seconds")
    time.sleep(max(0, (chunk_audio.shape[-1] - cfg.discarded_frames)*cfg.feat_len_sec - 
                    (loop_end_time - loop_start_time) - time_diff * cfg.finetune_realtime_ratio))


def write_seglst_file(seglst_dict_list, output_path):
    if len(seglst_dict_list) == 0:
        raise ValueError("seglst_dict_list is empty. No transcriptions were generated.")
    with open(output_path, 'w') as f:
        f.write(json.dumps(seglst_dict_list, indent=4) + '\n')
    logging.info(f"Saved the transcriptions of the streaming inference in\n:{output_path}")

def launch_serial_streaming(
    cfg, 
    asr_model, 
    diar_model, 
    streaming_buffer, 
    pad_and_drop_preencoded=False,
):
    streaming_buffer_iter = iter(streaming_buffer)

    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    feat_frame_count = 0
    
    session_start_time = time.time()
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded)
        loop_start_time = time.time()
        with torch.inference_mode():
            with autocast:
                with torch.no_grad(): 
                    multispk_asr_streamer.perform_serial_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )

        feat_frame_count += (chunk_audio.shape[-1] - cfg.discarded_frames)
        if cfg.real_time_mode:
            add_delay_for_real_time(cfg, 
                                    chunk_audio=chunk_audio,
                                    session_start_time=session_start_time,
                                    feat_frame_count=feat_frame_count,
                                    loop_end_time=time.time(),
                                    loop_start_time=loop_start_time
                                )
    return multispk_asr_streamer

def launch_parallel_streaming(
    cfg, 
    asr_model, 
    diar_model, 
    streaming_buffer, 
    pad_and_drop_preencoded=False,
    ):
    streaming_buffer_iter = iter(streaming_buffer)
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        # logging.info(f"Step ID: {step_num}")
        with torch.inference_mode():
            with autocast:
                with torch.no_grad(): 
                    drop_extra_pre_encoded = calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded)
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )
    return multispk_asr_streamer


@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:
    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)
        
    if cfg.diar_model_path is None and cfg.diar_pretrained_name is None:
        raise ValueError("Both cfg.diar_model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_file is None and cfg.manifest_file is None:
        raise ValueError("Both cfg.audio_file and cfg.manifest_file cannot be None!")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.diar_model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.diar_model_path, 
                                                                  map_location=map_location, strict=False)
    elif cfg.diar_model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model_path, 
                                                          map_location=map_location)
    else:
        raise ValueError("cfg.diar_model_path must end with.ckpt or.nemo!")
    
    # Model setup for inference 
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    diar_model._cfg.test_ds.manifest_filepath = cfg.manifest_file
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    diar_model = diar_model.eval()
    
    # Steaming mode setup
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_modules.chunk_len = cfg.chunk_len
    diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
    diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
    diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    diar_model.sortformer_modules.log = cfg.log
    diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate
    
    if cfg.audio_file is not None and cfg.manifest_file is not None:
        logging.warning("Both audio_file and manifest_file are specified. audio_file will be used with top priority.")
        input_type = "audio_file"
    elif cfg.audio_file is not None:
        logging.info("audio_file is specified. Using audio_file as input.")
        input_type = "audio_file"
    elif cfg.manifest_file is not None:
        logging.info("manifest_file is specified. Using manifest_file as input.")
        input_type = "manifest_file"
    else:
        raise ValueError("One of audio_file or manifest_file must be specified!")

    if cfg.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {cfg.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=cfg.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {cfg.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.asr_model)

    logging.info(asr_model.encoder.streaming_cfg)
    if cfg.set_decoder is not None:
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=cfg.set_decoder)
        else:
            raise ValueError("Decoder cannot get changed for non-Hybrid ASR models.")

    if cfg.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=cfg.use_amp)
    
    # Initialize to avoid "possibly used before assignment" error
    multispk_asr_streamer = None

    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(cfg.device)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. 
    # For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if cfg.chunk_size > 0:
        if cfg.shift_size < 0:
            shift_size = cfg.chunk_size
        else:
            shift_size = cfg.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=cfg.chunk_size, left_chunks=cfg.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the 
    # whole audio at the beginning When online_normalization is enabled, the normalization 
    # of the input features (mel-spectrograms) are done per step It is suggested to train 
    # the streaming models without any normalization in the input features.
    if cfg.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has"
                "no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False
    
    if cfg.audio_file is not None:
        # Stream a single audio file
        samples = [{'audio_filepath': cfg.audio_file,}]
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=asr_model,
            online_normalization=online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        cfg.batch_size = len(samples)
        streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)
        if cfg.parallel_speaker_strategy:
            multispk_asr_streamer = launch_serial_streaming(
                cfg=cfg,
                asr_model=asr_model,
                diar_model=diar_model,
                streaming_buffer=streaming_buffer,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,       
            )

        else:
            multispk_asr_streamer = launch_serial_streaming(
                cfg=cfg,
                asr_model=asr_model,
                diar_model=diar_model,
                streaming_buffer=streaming_buffer,
            )
    else:
        # Stream audio files in a manifest file in batched mode
        feat_per_sec = round(asr_model.cfg.preprocessor.window_stride * asr_model.cfg.encoder.subsampling_factor, 2)
        samples, rttms_mask_mats = get_multi_talker_samples_from_manifest(cfg, manifest_file=cfg.manifest_file, feat_per_sec=feat_per_sec, max_spks=cfg.max_num_of_spks)
        cfg.batch_size = len(samples)
        # Note: rttms_mask_mats contains PyTorch tensors, so we pass it directly instead of storing in config
        if cfg.spk_supervision == "rttm":
            diar_model.add_rttms_mask_mats(rttms_mask_mats, device=asr_model.device)

        logging.info(f"Loaded {len(samples)} from the manifest at {cfg.manifest_file}.")

        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=asr_model,
            online_normalization=online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        
        for sample_idx, sample in enumerate(samples):
            streaming_buffer.append_audio_file(sample['audio_filepath'], stream_id=-1)
            logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % cfg.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                if cfg.parallel_speaker_strategy:
                    multispk_asr_streamer = launch_parallel_streaming(
                        cfg=cfg,
                        asr_model=asr_model,
                        diar_model=diar_model,
                        streaming_buffer=streaming_buffer,
                        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                    )
                else:
                    multispk_asr_streamer = launch_serial_streaming(
                        cfg=cfg,
                        asr_model=asr_model,
                        diar_model=diar_model,
                        streaming_buffer=streaming_buffer,
                    )
                streaming_buffer.reset_buffer() 

    if cfg.output_path is not None and multispk_asr_streamer is not None:
        if cfg.parallel_speaker_strategy:
            multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
            write_seglst_file(seglst_dict_list=multispk_asr_streamer.instance_manager.seglst_dict_list, 
                              output_path=cfg.output_path)
        else:
            multispk_asr_streamer.generate_seglst_dicts_from_serial_streaming(samples=samples)
            write_seglst_file(seglst_dict_list=multispk_asr_streamer.instance_manager.seglst_dict_list, 
                              output_path=cfg.output_path)

if __name__ == '__main__':
    main()
