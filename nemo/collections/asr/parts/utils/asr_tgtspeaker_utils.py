import os
import re
import copy
import math
import random
import logging
import itertools
from copy import deepcopy
import concurrent.futures
from cytoolz import groupby
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import soundfile
from tqdm import tqdm
from scipy.stats import norm
from nltk.tokenize import SyllableTokenizer

import torch.utils.data
from lhotse.cut.set import mix
from lhotse.lazy import LazyJsonlIterator
from lhotse.cut import Cut, CutSet, MixedCut, MonoCut, MixTrack
from lhotse import SupervisionSet, SupervisionSegment, dill_enabled, AudioSource, Recording
from lhotse.utils import uuid4, compute_num_samples
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_hidden_length_from_sample_length,
    find_segments_from_rttm
)

def mix_noise(
    cuts,
    noise_manifests,
    snr,
    mix_prob,
):
    
    mixed_cuts = []
    assert 0.0 <= mix_prob <= 1.0, "mix_prob must be between 0.0 and 1.0"
    for cut in cuts:
        if random.uniform(0.0, 1.0) > mix_prob or cut.duration == 0:
            mixed_cuts.append(cut)
            continue
        to_mix_manifest = random.choice(noise_manifests)
        to_mix_cut = json_to_cut(to_mix_manifest)
        to_mix_cut = to_mix_cut.resample(16000)
        snr = random.uniform(*snr) if isinstance(snr, (list, tuple)) else snr
        mixed = cut.mix(to_mix_cut, snr = snr)
        mixed = mixed.truncate(duration=cut.duration)
        mixed_cuts.append(mixed) 
    return CutSet.from_cuts(mixed_cuts)

def rir_augment(cuts, prob):
    mixed_cuts = []
    for cut in cuts:
        if random.uniform(0.0, 1.0) > prob:
            mixed_cuts.append(cut)
        else:
            mixed_cuts.append(cut.reverb_rir())
    return CutSet.from_cuts(mixed_cuts)

def codec_augment(cuts, prob):
    mixed_cuts = []
    for cut in cuts:
        if random.uniform(0.0, 1.0) > prob:
            mixed_cuts.append(cut)
        else:
            mixed_cuts.append(cut.narrowband('mulaw'))
    return CutSet.from_cuts(mixed_cuts)


def speaker_to_target_w_query(
        a_cut, 
        query,
        add_separater_audio: bool = True,
        separater_duration: int = 1,
        num_speakers: int = 4, 
        num_sample_per_mel_frame: int = 160, 
        num_mel_frame_per_asr_frame: int = 8, 
        spk_tar_all_zero: bool = False, 
        boundary_segments: bool = False):
    '''
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor): set to True gives all zero "mask"
        boundary_segments (bool): set to True to include segments containing the boundary of the cut, False by default for multi-speaker ASR training
    
    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{cut}: only MixedCut and MonoCut are supported")
    segments_total = []
    for i, cut in enumerate(cut_list):
        if hasattr(cut, 'rttm_filepath') and cut.rttm_filepath is not None:
            rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        elif hasattr(cut, 'speaker_id') and cut.speaker_id is not None:
            rttms = SupervisionSet.from_segments([SupervisionSegment(
                id=uuid4(),
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                channel=1,
                speaker=cut.speaker_id,
                language=None
            )])
        else:
            raise ValueError(f"Cut {cut.id} does not have rttm_filepath or speaker_id")
        if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end, tolerance=0.0)
        else: # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)

        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)
    segments_total.sort(key = lambda rttm_sup: rttm_sup.start)
    seen = set()
    seen_add = seen.add
    if isinstance(a_cut, MixedCut):
        cut = a_cut
    if 'query_speaker_id' in cut.custom:
        speaker_lst = [cut.query_speaker_id] + [s.speaker for s in segments_total] #add query speaker as the first speaker
    else:
        speaker_lst = [s.speaker for s in segments_total]

    speaker_ats = [s for s in speaker_lst if not (s in seen or seen_add(s))]
    
    speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(speaker_ats)
    }
        #initialize mask matrices (num_speaker, encoder_hidden_len)
    if add_separater_audio:
        encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  query.num_samples + separater_duration * query.sampling_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

        separater_hidden_len = get_hidden_length_from_sample_length(separater_duration * query.sampling_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

        query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0

        mask = np.zeros((num_speakers, encoder_hidden_len))

        if hasattr(query, 'rttm_filepath') and query.rttm_filepath is not None:
            query_rttms = SupervisionSet.from_rttm(query.rttm_filepath)
            query_segments_iterator = find_segments_from_rttm(recording_id=query.recording_id, rttms=query_rttms, start_after=query.start, end_before=query.end, tolerance=0.0)
            query_segments_total = []
            for seg in query_segments_iterator:
                if seg.start < 0:
                    seg.duration += seg.start
                    seg.start = 0
                if seg.end > query.duration:
                    seg.duration -= seg.end - query.duration
                query_segments_total.append(seg)
            for rttm_sup in query_segments_total:
                    st = (
                                compute_num_samples(rttm_sup.start, query.sampling_rate)
                                if rttm_sup.start > 0
                                else 0
                            )
                    et = (
                                compute_num_samples(rttm_sup.end, query.sampling_rate)
                                if rttm_sup.end < query.duration
                                else compute_num_samples(query.duration, query.sampling_rate)
                            ) 
                    st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    mask[0, st_encoder_loc: et_encoder_loc] = 1
        else:
            mask[0,:query_hidden_len] = 1                

        for rttm_sup in segments_total:
            speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
            #only consider the first <num_speakers> speakers
            if speaker_idx < 4:
                st = (
                            compute_num_samples(rttm_sup.start, cut.sampling_rate)
                            if rttm_sup.start > 0
                            else 0
                        )
                et = (
                            compute_num_samples(rttm_sup.end, cut.sampling_rate)
                            if rttm_sup.end < cut.duration
                            else compute_num_samples(cut.duration, cut.sampling_rate)
                        )                   
                
                #map start time (st) and end time (et) to encoded hidden location
                st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                mask[speaker_idx, query_hidden_len + separater_hidden_len + st_encoder_loc: query_hidden_len + separater_hidden_len + et_encoder_loc] = 1

    else:
        encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
        query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0
        mask = np.zeros((num_speakers, encoder_hidden_len))

        if hasattr(query, 'rttm_filepath') and query.rttm_filepath is not None:
            query_rttms = SupervisionSet.from_rttm(query.rttm_filepath)
            query_segments_iterator = find_segments_from_rttm(recording_id=query.recording_id, rttms=query_rttms, start_after=query.start, end_before=query.end, tolerance=0.0)
            query_segments_total = []
            for seg in query_segments_iterator:
                if seg.start < 0:
                    seg.duration += seg.start
                    seg.start = 0
                if seg.end > query.duration:
                    seg.duration -= seg.end - query.duration
                query_segments_total.append(seg)
            for rttm_sup in query_segments_total:
                    st = (
                                compute_num_samples(rttm_sup.start, query.sampling_rate)
                                if rttm_sup.start > 0
                                else 0
                            )
                    et = (
                                compute_num_samples(rttm_sup.end, query.sampling_rate)
                                if rttm_sup.end < query.duration
                                else compute_num_samples(query.duration, query.sampling_rate)
                            ) 
                    st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    mask[0, st_encoder_loc: et_encoder_loc] = 1
        else:
            mask[0,:query_hidden_len] = 1  

        mask[0,:query_hidden_len] = 1

        for rttm_sup in segments_total:
            speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
            #only consider the first <num_speakers> speakers
            if speaker_idx < 4:
                st = (
                            compute_num_samples(rttm_sup.start, cut.sampling_rate)
                            if rttm_sup.start > 0
                            else 0
                        )
                et = (
                            compute_num_samples(rttm_sup.end, cut.sampling_rate)
                            if rttm_sup.end < cut.duration
                            else compute_num_samples(cut.duration, cut.sampling_rate)
                        )                   
                
                #map start time (st) and end time (et) to encoded hidden location
                st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                mask[speaker_idx, query_hidden_len + st_encoder_loc:query_hidden_len + et_encoder_loc] = 1

    return mask

def get_bounded_segment(start_time, total_duration, min_duration=1.0, max_duration=10.0):
    """
    Generate a segment within an audio clip with bounded duration.
    
    Args:
        start_time (float): Start time of the audio in seconds
        total_duration (float): Total duration of the audio in seconds
        min_duration (float): Minimum allowed segment duration in seconds
        max_duration (float): Maximum allowed segment duration in seconds
    
    Returns:
        tuple: (segment_start, segment_duration)
    """
    import random
    # Ensure max_duration doesn't exceed total_duration
    max_duration = min(max_duration, total_duration)
    
    # Ensure min_duration is not greater than max_duration
    min_duration = min(min_duration, max_duration)
    
    # Generate random duration within bounds
    segment_duration = np.round(random.uniform(min_duration, max_duration), decimals=3)
    
    # Calculate maximum possible start time
    max_start = total_duration - segment_duration
    
    # Generate random start time
    segment_start = np.round(random.uniform(start_time, start_time + max_start), decimals=3)
    
    return segment_start, segment_duration

def get_separator_audio(freq, sr, duration, ratio):
    # Generate time values
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Generate sine wave
    y = np.sin(2 * np.pi * freq * t) * 0.1

    y[:int(sr * duration * ratio )] = 0
    y[-int(sr * duration * ratio ):] = 0
    return y

def get_query_cut(cut):
    '''
    Extract query from the cut and saved as a separate cut

    Args:
        cut: An audio cut. The cut should contain keys "query_audio_filepath", "query_offet", "query_duration"

    Returns:
        query_cut: a cut containing query information
    '''    
    if 'query_audio_filepath' in cut.custom:
        #no query is provided for query cut
        query_rec = Recording.from_file(cut.query_audio_filepath)
        if query_rec.sampling_rate != 16000:
            query_rec = query_rec.resample(sampling_rate=16000)
        query_sups = [SupervisionSegment(id=query_rec.id+'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration), recording_id = query_rec.id, start = 0, duration = cut.query_duration, speaker = cut.query_speaker_id)]
        #additional information for query
        custom = {
            'rttm_filepath' : cut.custom.get('query_rttm_filepath', None)
        }
        query_cut = MonoCut(id = query_rec.id +'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration),
                            start = cut.query_offset,
                            duration = cut.query_duration,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
        query_cut.custom = custom
        return query_cut
    else:
        query_rec = cut.recording
        query_sups = [SupervisionSegment(id=cut.id+'_query_dummy', recording_id = query_rec.id, start = 0, duration = 0, speaker = None)]
        query_cut = MonoCut(id = cut.id +'_query_no_ts_'+str(cut.start)+'_'+str(cut.duration),
                            start = 0,
                            duration = 0,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
        return query_cut
    
def json_to_cut(json_dict):
    """
    Convert a json dictionary to a Cut instance.
    """
    audio_path = json_dict["audio_filepath"]
    duration = json_dict["duration"]
    offset = json_dict.get("offset", None)
    cut = _create_cut(
        audio_path=audio_path, offset=offset, duration=duration, sampling_rate=json_dict.get("sampling_rate", None)
    )
    # Note that start=0 and not start=offset because supervision's start if relative to the
    # start of the cut; and cut.start is already set to offset
    cut.supervisions.append(
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=cut.duration,
            text=json_dict.get("text"),
            language=json_dict.get("language", "en"),
        )
    )
    cut.custom = json_dict

    return cut

def _create_cut(
    audio_path: str,
    offset: float,
    duration: float,
    sampling_rate: int | None = None,
) -> Cut:
    
    recording = _create_recording(audio_path, duration, sampling_rate)
    cut = recording.to_cut()
    if offset is not None:
        cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
        cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
    return cut

def _create_recording(
    audio_path: str,
    duration: float,
    sampling_rate: int | None = None,
) -> Recording:
    if sampling_rate is not None:
        # TODO(pzelasko): It will only work with single-channel audio in the current shape.
        return Recording(
            id=audio_path,
            sources=[AudioSource(type="file", channels=[0], source=audio_path)],
            sampling_rate=sampling_rate,
            num_samples=compute_num_samples(duration, sampling_rate),
            duration=duration,
            channel_ids=[0],
        )
    else:
        return Recording.from_file(audio_path)

class LibriSpeechMixGenerator_tgt():
    def __init__(self):
        pass

    def generate(self, cuts):
        cut_set = []
        for cut in tqdm(cuts, desc=f"Generating speaker intra-session mixtures", ncols=128):
            offsets = cut.delays
            durations = cut.durations
            wavs = cut.wavs
            text = cut.text
            query_audio_filepath = cut.query_audio_filepath
            query_speaker_id = cut.query_speaker_id
            query_offset = cut.query_offset
            query_duration = cut.query_duration
            rttm_filepath = cut.rttm_filepath
            # speakers = cut.speakers

            tracks = []
            for i, (offset, duration, wav) in enumerate(zip(offsets, durations, wavs)):
                custom = {
                        'pnc': 'no',
                        'source_lang': 'en',
                        'target_lang': 'en',
                        'task': 'asr',
                        'speaker_id': wav.split('/')[-1].split('-')[0]
                    }
                wav_dur = soundfile.info(wav).duration
                wav_samples = soundfile.info(wav).frames
                cut_1spk = MonoCut(
                    id=wav.split('/')[-1].replace('.wav', ''),
                    start=0,
                    duration=duration,
                    channel=0,
                    supervisions=[],
                    recording=Recording(
                        id=cut.rttm_filepath.split('/')[-1].replace('.rttm',''),
                        sources=[
                            AudioSource(
                                type='file',
                                channels=[0],
                                source=wav
                            )
                        ],
                        sampling_rate=16000, 
                        num_samples=wav_samples,
                        duration=wav_dur
                    ),
                    custom = custom
                )
                tracks.append(MixTrack(cut=deepcopy(cut_1spk), type=type(cut_1spk), offset=offset))

            mixed_cut = MixedCut(id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)
            #modify monocut's recording id for further rttm reading
            sup = SupervisionSegment(
                id= mixed_cut.id,
                recording_id=mixed_cut.id,
                start=0,
                duration=mixed_cut.duration,
                text=cut.text,
            )
            mixed_cut.tracks[0].cut.supervisions = [sup]
            custom = {
                    'query_audio_filepath': query_audio_filepath,
                    'query_speaker_id': query_speaker_id,
                    'query_offset': query_offset,
                    'query_duration': query_duration,
                    'rttm_filepath': rttm_filepath,
                    'custom': None
                }
            mixed_cut.tracks[0].cut.custom.update(custom)
            cut_set.append(mixed_cut)
        
        return CutSet.from_cuts(cut_set)
    

class LibriSpeechMixSimulator_tgt():

    def __init__(
        self,
        data_type: str = "msasr",
        min_delay: float = 0.5,
        max_num_speakers: int = 4,
        speaker_count_distribution: List[float] = [0, 2, 3, 4],
        query_duration: List[float] = [1, 10],
        delay_factor: int = 1
    ):
        """
        Args:
        data_type: the type of data to simulate. Either 'msasr', 'tsasr' or 'diar'. [Default: 'msasr']
        min_delay: the minimum delay between the segments. [Default: 0.5]
        max_num_speakers: the maximum number of speakers in the meeting. [Default: 4]
        speaker_token_position: the position of the speaker token in the text. Either 'sot', 'word', or 'segments'. [Default: 'sot']
        speaker_count_distribution: the speaker count distribution for the simulated meetings. [Default: [0, 2, 3, 4]]
        query_duration: the min and max query duration for the simulated meetings. [Default: [1, 10]]
        delay_factor: the number of times to repeat the meeting with the same speakers. [Default: 1]
        """
        super().__init__()
        self.data_type = data_type
        self.min_delay = min_delay
        self.delay_factor = delay_factor
        self.max_num_speakers = max_num_speakers
        self.speaker_count_distribution = speaker_count_distribution
        self.query_duration = query_duration
        assert len(speaker_count_distribution) == max_num_speakers, f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {max_num_speakers}"
        assert len(query_duration) == 2, f"set query duration to be [min, max] in s"

    def fit(self, cuts) -> CutSet:
        self.speaker_id2cut_ids = defaultdict(list)
        self.id2cuts = defaultdict(list)
        for cut in tqdm(cuts, desc="Reading segments", ncols=100):
            # if not hasattr(cut, 'dataset_id') or cut.dataset_id != 'librispeech':
            #     continue
            if hasattr(cuts[0], 'speaker_id'):
                speaker_id = cut.speaker_id
            else: #LibriSpeech
                speaker_id = cut.recording_id.split('-')[0]
                cut.speaker_id = speaker_id
            self.speaker_id2cut_ids[speaker_id].append(cut.id)
            self.id2cuts[cut.id] = cut
        
        self.speaker_ids = list(self.speaker_id2cut_ids.keys())

    def _create_mixture(self, n_speakers: int, non_query_sample = False) -> MixedCut:
        sampled_speaker_ids = random.sample(self.speaker_ids, n_speakers)
        
        mono_cuts = []
        cut_ids = []
        for speaker_id in sampled_speaker_ids:
            cut_id = random.choice(self.speaker_id2cut_ids[speaker_id])
            cut = self.id2cuts[cut_id]
            mono_cuts.append(cut)
            cut_ids.append(cut_id)

        mixed_cuts = []
        if n_speakers == 1:
            #do not add delay factor to single-spk sample as augmented one will be the same
            delay_factor = 1
        else:
            delay_factor = self.delay_factor
        for i in range(delay_factor):
            tracks = []
            offset = 0.0
            for mono_cut in mono_cuts:
                custom = {
                        'pnc': 'no',
                        'source_lang': 'en',
                        'target_lang': 'en',
                        'task': 'asr'
                    }
                mono_cut.custom.update(custom)
                tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
                offset += random.uniform(self.min_delay, mono_cut.duration)
        
            mixed_cut = MixedCut(id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)

            if self.data_type == "tsasr":
                if non_query_sample:
                    query_speaker_id = random.sample(set(self.speaker_ids) - set(sampled_speaker_ids), 1)[0]
                    query_cut_list = self.speaker_id2cut_ids[query_speaker_id]
                else:
                    index = random.randrange(len(sampled_speaker_ids))
                    query_speaker_id = sampled_speaker_ids[index]
                    query_cut_list = deepcopy(self.speaker_id2cut_ids[query_speaker_id])
                    query_cut_list.remove(cut_ids[index])
                if len(query_cut_list) == 0:
                    #no query utterance different from target utterance is found
                    return mixed_cuts
                query_id = random.choice(query_cut_list)
                query_cut = self.id2cuts[query_id]
                text = self.get_text(mixed_cut, query_speaker_id) if not non_query_sample else ""
                sup = SupervisionSegment(id = mixed_cut.id, recording_id = mixed_cut.id, start = 0, duration=mixed_cut.duration, text = text)
                query_offset, query_duration = self.get_bounded_segment(query_cut.start, query_cut.duration, min_duration=self.query_duration[0], max_duration=self.query_duration[1])
                custom = {
                        'pnc': 'no',
                        'source_lang': 'en',
                        'target_lang': 'en',
                        'task': 'asr',
                        'query_audio_filepath': query_cut.recording.sources[0].source,
                        'query_speaker_id': query_speaker_id,
                        'query_offset': query_offset,
                        'query_duration': query_duration,
                        'query_rttm_filepath': query_cut.rttm_filepath if hasattr(query_cut, 'rttm_filepath') else None,
                        'custom': None 
                    }
                mixed_cut.tracks[0].cut.supervisions = [sup]
                mixed_cut.tracks[0].cut.custom.update(custom)

            mixed_cuts.append(mixed_cut)

        return mixed_cuts
    
    # TODO: text is necessary for msasr and tsasr, but not for diar
    def get_text(self, cut: MixedCut, query_speaker_id) -> str:
        for i, track in enumerate(cut.tracks):
            if track.cut.speaker_id == query_speaker_id:
                return track.cut.text
        return ValueError ('Error in finding query speaker in target utterance')
    

    def apply_speaker_distribution(self, num_meetings: int, speaker_count_distribution) -> Dict[int, int]:
        """
        Balance the speaker distribution for the simulated meetings.
        Args:
            num_meetings: The total number of simulated meetings.
            speaker_count_distribution: The speaker count distribution for the simulated meetings.
        For each number of speakers, calculate the number of meetings needed to balance the distribution.
        """

        total_spk = sum(speaker_count_distribution)
        num_speakers2num_meetings = {}
        for i_spk in range(self.max_num_speakers):
            num_speakers2num_meetings[i_spk+1] = round(num_meetings * speaker_count_distribution[i_spk] / total_spk)

        return num_speakers2num_meetings
            
    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        non_existing_query_ratio: float = 0,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        self.fit(cuts)

        self.num_speakers2num_meetings = self.apply_speaker_distribution(num_meetings, self.speaker_count_distribution)

        cut_set = []
        for n_speakers, n_mt in self.num_speakers2num_meetings.items():
            if n_mt <= 0:
                continue
            for i in tqdm(range(n_mt), desc=f"Simulating {n_speakers}-speaker mixtures", ncols=128):
                cut_set.extend(self._create_mixture(n_speakers=n_speakers))
        if non_existing_query_ratio > 0:
            #add samples where query speaker not in target utterance and set text field to be empty straing
            num_no_query_samples = int(num_meetings * non_existing_query_ratio)
            for i in tqdm(range(num_no_query_samples), desc=f"Simulating non existing query samples", ncols=128):
                cut_set.extend(self._create_mixture(n_speakers=np.random.choice(np.arange(1, self.max_num_speakers+1)), non_query_sample=True))           
        return CutSet.from_cuts(cut_set).shuffle()


class TargetSpeakerSimulator():
    """
    This class is used to simulate target-speaker audio data,
    which can be used for target-speaker ASR and speaker diarization training.
    """
    def __init__(
        self, 
        manifest_filepath, 
        num_speakers, 
        simulator_type,
        min_delay=0.5,
        max_delay_after_each_mono: float = 0,
        non_query_sample: bool = False,
        query_duration: List[float] = [3, 10]
    ):
        """
        Args:
            manifest_filepath (str): The path to the manifest file.
            num_speakers (int): The number of speakers in the simulated audio.
            simulator_type (str): The type of simulator to use.
                - 'lsmix': LibriSpeechMix-style training sample (mix single speaker audio).
            min_delay (float): The minimum delay between speakers
                to avoid the same starting time for multiple speakers.
            max_delay_after_each_mono (float): The maximum delay of another mono cut after each mono cut. Default is 0, means audio mixtures guaranteed to overlap. 
            non_query_sample (bool): Whether to sample a sample where query speaker not in target audio. Default is False.
            query_duration (list): The duration of the query sample in s. Default is [3, 10].
            TODO: add mono_duration (list): Select random start and duration for each single speaker audio according to mono_duration [min max]. Emprically, need to set min_duration > 0 if max_after_each_mono > 0!!!
        """
    
        self.manifests = LazyJsonlIterator(manifest_filepath)
        self.min_delay = min_delay
        self.max_delay_after_each_mono = max_delay_after_each_mono
        self.num_speakers = num_speakers
        self.simulator_type = simulator_type
        self.query_duration = query_duration
        self.non_query_sample = non_query_sample

        self.spk2manifests = groupby(lambda x: x["speaker_id"], self.manifests)
        self.speaker_ids = list(self.spk2manifests.keys())

        if simulator_type == 'lsmix':    
            self.simulator = self.LibriSpeechMixSimulator_tgt
        elif simulator_type == 'meeting':
            #TODO
            raise NotImplementedError("MeetingSimulator is not implemented yet.")
        elif simulator_type == 'conversation':
            #TODO
            raise NotImplementedError("ConversationSimulator is not implemented yet.")

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.simulator()

    def LibriSpeechMixSimulator_tgt(self):
        """
        This function simulates a LibriSpeechMix-style TS-ASR training sample.
        Returns:
            mixed_cut: a mixed cut containing target-speaker audio and query speaker audio.
        """
        # Sample the speakers
        sampled_speaker_ids = random.sample(self.speaker_ids, self.num_speakers)
        
        # Create tracks for all speakers at once
        tracks = []
        offset = 0
        
        # Common custom dict to avoid recreating
        base_custom = {
            'pnc': 'no', 
            'source_lang': 'en',
            'target_lang': 'en',
            'task': 'asr'
        }

        # Create tracks in a single loop
        for speaker_id in sampled_speaker_ids:
            manifest = random.choice(self.spk2manifests[speaker_id])
            mono_cut = json_to_cut(manifest)
            mono_cut.custom.update(base_custom)
            tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
            offset += random.uniform(self.min_delay, mono_cut.duration + self.max_delay_after_each_mono)

        # Create mixed cut
        mixed_cut = MixedCut(
            id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()),
            tracks=tracks
        )

        # # Handle query speaker selection
        query_speaker_id = random.choice(list(set(self.speaker_ids) - set(sampled_speaker_ids))) if self.non_query_sample else random.choice(sampled_speaker_ids)

        # # Get query cut
        query_manifest = random.choice(self.spk2manifests[query_speaker_id])
        query_cut = json_to_cut(query_manifest)
        # query_speaker_id = mono_cut.speaker_id
        # query_cut = mono_cut

        # Create supervision
        text = self.get_text(mixed_cut, query_speaker_id) if not self.non_query_sample else ""
        sup = SupervisionSegment(
            id=mixed_cut.id,
            recording_id=mixed_cut.id,
            start=0,
            duration=mixed_cut.duration,
            text=text
        )

        # Get query segment bounds
        query_offset, query_duration = get_bounded_segment(
            query_cut.start,
            query_cut.duration,
            min_duration=self.query_duration[0],
            max_duration=self.query_duration[1]
        )

        # Update cut with final metadata
        custom = {
            **base_custom,
            'query_audio_filepath': query_cut.recording.sources[0].source,
            'query_speaker_id': query_speaker_id,
            'query_offset': query_offset,
            'query_duration': query_duration,
            'query_rttm_filepath': query_cut.rttm_filepath if hasattr(query_cut, 'rttm_filepath') else None,
            'custom': None
        }
        
        mixed_cut.tracks[0].cut.supervisions = [sup]
        mixed_cut.tracks[0].cut.custom.update(custom)

        return mixed_cut
    
    def get_text(self, cut: MixedCut, query_speaker_id) -> str:
        """
        Get the text of the query speaker in the target utterance.
        Args:
            cut (MixedCut): The mixed cut containing target-speaker audio and query speaker audio.
            query_speaker_id (str): The id of the query speaker.
        Returns:
            text (str): The text of the query speaker in the target utterance.
        """
        for i, track in enumerate(cut.tracks):
            if track.cut.speaker_id == query_speaker_id:
                return track.cut.text
        return ValueError ('Error in finding query speaker in target utterance')


    def MeetingSimulator(self):
        raise NotImplementedError("MeetingSimulator is not implemented yet.")   

    def ConversationSimulator(self):
        raise NotImplementedError("ConversationSimulator is not implemented yet.")
    
    # TODO: text is necessary for msasr and tsasr, but not for diar
