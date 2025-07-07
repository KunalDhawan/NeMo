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

"""
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `--pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

## Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by --set_decoder DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.

## Multi-lookahead models
For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size. To change it, you may use --att_context_size, for example --att_context_size [70,1].

## Cache Optimizations (Optional - baseline behavior unchanged)

To use cache optimizations for improved performance while maintaining accuracy:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=2048 \
    --use_true_circular_buffers \
    --use_optimized_cache \
    --enable_memory_pool \
    --cache_dtype=bfloat16 \
    --enable_mixed_precision_cache \
    --use_amp

Cache optimization parameters:
--use_true_circular_buffers: Enable true circular buffers for attention layers (eliminates concatenation operations)
--use_optimized_cache: Enable general cache optimizations including memory pool and efficient tensor operations  
--enable_memory_pool: Enable memory pool for efficient cache tensor reuse across streaming sessions
--enable_quantization: Enable cache quantization to reduce memory usage (may slightly affect accuracy)
--cache_dtype: Data type for cache tensors (float32/float16/bfloat16 for memory efficiency)
--pool_size_limit: Maximum number of tensors to keep in memory pool for reuse (default: 50)
--cache_cleanup_interval: Frequency of cache cleanup operations (default: 25)
--cache_memory_fraction: Fraction of GPU memory to allocate for caches (default: 0.4)
--enable_mixed_precision_cache: Use mixed precision for cache operations

Note: When no optimization flags are provided, the script uses baseline behavior identical to the original implementation.

## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be used as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_streaming_infer.py \
    --asr_model=stt_en_conformer_ctc_large \
    --chunk_size=100 \
    --shift_size=50 \
    --left_chunks=2 \
    --online_normalization \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

"""


import contextlib
import json
import os
import time
from argparse import ArgumentParser

import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.utils import logging


def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(
    asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False
):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same
        with torch.inference_mode():
            with autocast:
                processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
                with torch.no_grad():
                    (
                        pred_out_offline,
                        transcribed_texts,
                        cache_last_channel_next,
                        cache_last_time_next,
                        cache_last_channel_len,
                        best_hyp,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        return_transcription=True,
                    )
        final_offline_tran = extract_transcriptions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        with torch.inference_mode():
            with autocast:
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped

                with torch.no_grad():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = asr_model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                            asr_model, step_num, pad_and_drop_preencoded
                        ),
                        return_transcription=True,
                    )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
        # Normally they should be exactly the same predictions for streaming models
        pred_out_stream_cat = torch.cat(pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )

    return final_streaming_tran, final_offline_tran


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model",
        type=str,
        required=True,
        help="Path to an ASR model .nemo file or name of a pretrained model.",
    )
    parser.add_argument(
        "--device", type=str, help="The device to load the model onto and perform the streaming", default="cuda"
    )
    parser.add_argument("--audio_file", type=str, help="Path to an audio file to perform streaming", default=None)
    parser.add_argument(
        "--manifest_file",
        type=str,
        help="Path to a manifest file containing audio files to perform streaming",
        default=None,
    )
    parser.add_argument("--use_amp", action="store_true", help="Whether to use AMP")
    parser.add_argument("--debug_mode", action="store_true", help="Whether to print more detail in the output.")
    parser.add_argument(
        "--compare_vs_offline",
        action="store_true",
        help="Whether to compare the output of the model with the offline mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to be used to perform streaming in batch mode with multiple streams",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=-1,
        help="The chunk_size to be used for models trained with full context and offline models",
    )
    parser.add_argument(
        "--shift_size",
        type=int,
        default=-1,
        help="The shift_size to be used for models trained with full context and offline models",
    )
    parser.add_argument(
        "--left_chunks",
        type=int,
        default=2,
        help="The number of left chunks to be used as left context via caching for offline models",
    )

    parser.add_argument(
        "--online_normalization",
        default=False,
        action='store_true',
        help="Perform normalization on the run per chunk.",
    )
    parser.add_argument(
        "--output_path", type=str, help="path to output file when manifest is used as input", default=None
    )
    parser.add_argument(
        "--pad_and_drop_preencoded",
        action="store_true",
        help="Enables padding the audio input and then dropping the extra steps after the pre-encoding for all the steps including the the first step. It may make the outputs of the downsampling slightly different from offline mode for some techniques like striding or sw_striding.",
    )

    parser.add_argument(
        "--set_decoder",
        choices=["ctc", "rnnt"],
        default=None,
        help="Selects the decoder for Hybrid ASR models which has both the CTC and RNNT decoder. Supported decoders are ['ctc', 'rnnt']",
    )

    parser.add_argument(
        "--att_context_size",
        type=str,
        default=None,
        help="Sets the att_context_size for the models which support multiple lookaheads",
    )

    parser.add_argument(
        "--matmul-precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
        help="Set torch matmul precision",
    )

    parser.add_argument("--strategy", type=str, default="greedy_batch", help="decoding strategy to use")

    # ========================
    # Cache Optimization Parameters (Optional - baseline behavior unchanged when not used)
    # ========================
    parser.add_argument(
        "--use_true_circular_buffers",
        action="store_true",
        default=False,
        help="Enable true circular buffers for attention layers (eliminates concatenation operations for better performance)"
    )
    
    parser.add_argument(
        "--use_optimized_cache",
        action="store_true", 
        default=False,
        help="Enable general cache optimizations including memory pool and efficient tensor operations"
    )
    
    parser.add_argument(
        "--enable_memory_pool",
        action="store_true",
        default=False,
        help="Enable memory pool for efficient cache tensor reuse across streaming sessions"
    )
    
    parser.add_argument(
        "--enable_quantization",
        action="store_true",
        default=False,
        help="Enable cache quantization to reduce memory usage (may slightly affect accuracy)"
    )
    
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for cache tensors (float16/bfloat16 can reduce memory usage)"
    )
    
    parser.add_argument(
        "--pool_size_limit",
        type=int,
        default=50,
        help="Maximum number of tensors to keep in memory pool for reuse"
    )
    
    parser.add_argument(
        "--cache_cleanup_interval",
        type=int,
        default=25,
        help="Frequency of cache cleanup operations to manage memory usage"
    )
    
    parser.add_argument(
        "--cache_memory_fraction",
        type=float,
        default=0.4,
        help="Fraction of GPU memory to allocate for caches (0.1-0.8 recommended)"
    )
    
    parser.add_argument(
        "--enable_mixed_precision_cache",
        action="store_true",
        default=False,
        help="Use mixed precision for cache operations to improve performance"
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.matmul_precision)
    if (args.audio_file is None and args.manifest_file is None) or (
        args.audio_file is not None and args.manifest_file is not None
    ):
        raise ValueError("One of the audio_file and manifest_file should be non-empty!")

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)

    logging.info(asr_model.encoder.streaming_cfg)
    if args.set_decoder is not None:
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=args.set_decoder)
        else:
            raise ValueError("Decoder cannot get changed for non-Hybrid ASR models.")

    if args.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=json.loads(args.att_context_size))
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=args.use_amp)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    # ========================
    # Configure Cache Optimizations IMMEDIATELY after model loading (ONLY when explicitly enabled - baseline unaffected)
    # ========================
    optimization_flags_enabled = any([
        args.use_true_circular_buffers, 
        args.use_optimized_cache, 
        args.enable_memory_pool, 
        args.enable_quantization, 
        args.enable_mixed_precision_cache
    ])
    
    if optimization_flags_enabled:
        logging.info("Cache optimizations explicitly enabled - configuring optimized inference...")
        
        # Convert cache_dtype string to torch dtype
        cache_dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        cache_dtype = cache_dtype_map[args.cache_dtype]
        
        # Enable cache optimization on the encoder
        if hasattr(asr_model.encoder, 'enable_cache_optimization'):
            asr_model.encoder.enable_cache_optimization(
                enable=args.use_optimized_cache,
                enable_quantization=args.enable_quantization
            )
            logging.info(f"Enabled cache optimization: {args.use_optimized_cache}, quantization: {args.enable_quantization}")
        
        # CRITICAL FIX: Configure streaming settings IMMEDIATELY after model loading
        if hasattr(asr_model.encoder, 'streaming_cfg'):
            # Debug: Check what type of object streaming_cfg is
            print(f"[DEBUG] EARLY: Reading from streaming_cfg: {type(asr_model.encoder.streaming_cfg)}")
            print(f"[DEBUG] EARLY: streaming_cfg attributes: {dir(asr_model.encoder.streaming_cfg)}")
            
            # Direct attribute assignment (CacheAwareStreamingConfig is a dataclass, not OmegaConf)
            try:
                # Try to set attributes directly
                streaming_cfg = asr_model.encoder.streaming_cfg
                
                # Debug: Check current value before setting
                current_value = getattr(streaming_cfg, 'use_true_circular_buffers', 'NOT_FOUND')
                print(f"[DEBUG] EARLY: Before setting - use_true_circular_buffers: {current_value}")
                print(f"[DEBUG] EARLY: Setting use_true_circular_buffers to: {args.use_true_circular_buffers}")
                
                # Set cache optimization attributes directly
                if not hasattr(streaming_cfg, 'use_true_circular_buffers'):
                    setattr(streaming_cfg, 'use_true_circular_buffers', args.use_true_circular_buffers)
                else:
                    streaming_cfg.use_true_circular_buffers = args.use_true_circular_buffers
                    
                # Debug: Check value after setting
                new_value = getattr(streaming_cfg, 'use_true_circular_buffers', 'NOT_FOUND')
                print(f"[DEBUG] EARLY: After setting - use_true_circular_buffers: {new_value}")
                    
                if not hasattr(streaming_cfg, 'use_optimized_cache'):
                    setattr(streaming_cfg, 'use_optimized_cache', args.use_optimized_cache)
                else:
                    streaming_cfg.use_optimized_cache = args.use_optimized_cache
                    
                if not hasattr(streaming_cfg, 'enable_memory_pool'):
                    setattr(streaming_cfg, 'enable_memory_pool', args.enable_memory_pool)
                else:
                    streaming_cfg.enable_memory_pool = args.enable_memory_pool
                    
                if not hasattr(streaming_cfg, 'enable_quantization'):
                    setattr(streaming_cfg, 'enable_quantization', args.enable_quantization)
                else:
                    streaming_cfg.enable_quantization = args.enable_quantization
                    
                if not hasattr(streaming_cfg, 'cache_dtype'):
                    setattr(streaming_cfg, 'cache_dtype', cache_dtype)
                else:
                    streaming_cfg.cache_dtype = cache_dtype
                    
                if not hasattr(streaming_cfg, 'pool_size_limit'):
                    setattr(streaming_cfg, 'pool_size_limit', args.pool_size_limit)
                else:
                    streaming_cfg.pool_size_limit = args.pool_size_limit
                    
                if not hasattr(streaming_cfg, 'cache_cleanup_interval'):
                    setattr(streaming_cfg, 'cache_cleanup_interval', args.cache_cleanup_interval)
                else:
                    streaming_cfg.cache_cleanup_interval = args.cache_cleanup_interval
                    
                if not hasattr(streaming_cfg, 'cache_memory_fraction'):
                    setattr(streaming_cfg, 'cache_memory_fraction', args.cache_memory_fraction)
                else:
                    streaming_cfg.cache_memory_fraction = args.cache_memory_fraction
                    
                if not hasattr(streaming_cfg, 'enable_mixed_precision_cache'):
                    setattr(streaming_cfg, 'enable_mixed_precision_cache', args.enable_mixed_precision_cache)
                else:
                    streaming_cfg.enable_mixed_precision_cache = args.enable_mixed_precision_cache
                    
                print(f"[DEBUG] EARLY: Successfully configured streaming_cfg with optimization parameters")
                
                # Configure circular buffers for attention layers IMMEDIATELY
                if args.use_true_circular_buffers and hasattr(asr_model.encoder, 'layers'):
                    logging.info("EARLY: Configuring circular buffers for attention layers...")
                    for layer_idx, layer in enumerate(asr_model.encoder.layers):
                        if hasattr(layer, 'set_circular_buffer_config'):
                            layer.set_circular_buffer_config(
                                use_circular_buffers=args.use_true_circular_buffers,
                                optimization_enabled=args.use_optimized_cache
                            )
                            logging.info(f"EARLY: Configured circular buffers for layer {layer_idx}")
                
            except Exception as e:
                print(f"[WARNING] EARLY: Failed to configure streaming_cfg: {e}")
                print(f"[DEBUG] EARLY: Using BASELINE caching (no optimizations)")
        
        # Log the optimization configuration
        optimization_summary = {
            "use_true_circular_buffers": args.use_true_circular_buffers,
            "use_optimized_cache": args.use_optimized_cache,
            "enable_memory_pool": args.enable_memory_pool,
            "enable_quantization": args.enable_quantization,
            "cache_dtype": args.cache_dtype,
            "pool_size_limit": args.pool_size_limit,
            "cache_cleanup_interval": args.cache_cleanup_interval,
            "cache_memory_fraction": args.cache_memory_fraction,
            "enable_mixed_precision_cache": args.enable_mixed_precision_cache
        }
        logging.info(f"EARLY: Optimized cache configuration: {optimization_summary}")
    else:
        # Baseline behavior - no modifications to the original code path
        logging.info("Using baseline inference (no optimization flags detected)")

    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = args.strategy
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.fused_batch_size = -1
            if not (max_symbols := decoding_cfg.greedy.get("max_symbols")) or max_symbols <= 0:
                decoding_cfg.greedy.max_symbols = 10
        if hasattr(asr_model, "cur_decoder"):
            # hybrid model, explicitly pass decoder type, otherwise it will be set to "rnnt"
            asr_model.change_decoding_strategy(decoding_cfg, decoder_type=asr_model.cur_decoder)
        else:
            asr_model.change_decoding_strategy(decoding_cfg)

    # chunk_size is set automatically for models trained for streaming. For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if args.chunk_size > 0:
        if args.shift_size < 0:
            shift_size = args.chunk_size
        else:
            shift_size = args.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the whole audio at the beginning
    # When online_normalization is enabled, the normalization of the input features (mel-spectrograms) are done per step
    # It is suggested to train the streaming models without any normalization in the input features.
    if args.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )
    if args.audio_file is not None:
        # stream a single audio file
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
        perform_streaming(
            asr_model=asr_model,
            streaming_buffer=streaming_buffer,
            compare_vs_offline=args.compare_vs_offline,
            pad_and_drop_preencoded=args.pad_and_drop_preencoded,
        )
    else:
        # stream audio files in a manifest file in batched mode
        samples = []
        all_streaming_tran = []
        all_offline_tran = []
        all_refs_text = []

        with open(args.manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)

        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        start_time = time.time()
        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], stream_id=-1
            )
            if "text" in sample:
                all_refs_text.append(sample["text"])
            logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                streaming_tran, offline_tran = perform_streaming(
                    asr_model=asr_model,
                    streaming_buffer=streaming_buffer,
                    compare_vs_offline=args.compare_vs_offline,
                    debug_mode=args.debug_mode,
                    pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                )
                all_streaming_tran.extend(streaming_tran)
                if args.compare_vs_offline:
                    all_offline_tran.extend(offline_tran)
                streaming_buffer.reset_buffer()

        if args.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")

        end_time = time.time()
        logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

        # stores the results including the transcriptions of the streaming inference in a json file
        if args.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
            fname = (
                "streaming_out_"
                + os.path.splitext(os.path.basename(args.asr_model))[0]
                + "_"
                + os.path.splitext(os.path.basename(args.manifest_file))[0]
                + ".json"
            )

            hyp_json = os.path.join(args.output_path, fname)
            os.makedirs(args.output_path, exist_ok=True)
            with open(hyp_json, "w") as out_f:
                for i, hyp in enumerate(all_streaming_tran):
                    record = {
                        "pred_text": hyp,
                        "text": all_refs_text[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    }
                    out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
