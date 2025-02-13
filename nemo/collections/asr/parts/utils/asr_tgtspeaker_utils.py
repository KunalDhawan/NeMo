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
from lhotse.cut import CutSet, MixedCut, MonoCut, MixTrack
from lhotse import SupervisionSet, SupervisionSegment, dill_enabled, AudioSource, Recording
from lhotse.utils import uuid4

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
        query_rec = Recording.from_file(cut.query_audio_filepath)
        if query_rec.sampling_rate != 16000:
            query_rec = query_rec.resample(sampling_rate=16000)
        query_sups = [SupervisionSegment(id=query_rec.id+'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration), recording_id = query_rec.id, start = 0, duration = cut.query_duration, speaker = cut.query_speaker_id)]
        query_cut = MonoCut(id = query_rec.id +'_query'+str(cut.query_offset)+'-'+str(cut.query_offset + cut.query_duration),
                            start = cut.query_offset,
                            duration = cut.query_duration,
                            channel = 0,
                            recording = query_rec,
                            supervisions = query_sups)
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

    def _create_mixture(self, n_speakers: int) -> MixedCut:
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
                index = random.randrange(len(sampled_speaker_ids))
                query_speaker_id = sampled_speaker_ids[index]
                query_cut_list = deepcopy(self.speaker_id2cut_ids[query_speaker_id])
                query_cut_list.remove(cut_ids[index])
                if len(query_cut_list) == 0:
                    #no query utterance different from target utterance is found
                    return mixed_cuts
                query_id = random.choice(query_cut_list)
                query_cut = self.id2cuts[query_id]
                text = self.get_text(mixed_cut, query_speaker_id)
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
                        'custom': None 
                    }
                mixed_cut.tracks[0].cut.supervisions = [sup]
                mixed_cut.tracks[0].cut.custom.update(custom)

            mixed_cuts.append(mixed_cut)

        return mixed_cuts
    
    def _create_non_existing_query_mixture(self, n_speakers: int) -> MixedCut:
        sampled_speaker_ids = random.sample(self.speaker_ids, n_speakers)
        
        mono_cuts = []
        cut_ids = []
        for speaker_id in sampled_speaker_ids:
            cut_id = random.choice(self.speaker_id2cut_ids[speaker_id])
            cut = self.id2cuts[cut_id]
            mono_cuts.append(cut)
            cut_ids.append(cut_id)

        mixed_cuts = []
        for i in range(self.delay_factor):
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
                # index = random.randrange(len(sampled_speaker_ids))
                query_speaker_id = random.sample(set(self.speaker_ids) - set(sampled_speaker_ids), 1)[0]
                query_cut_list = self.speaker_id2cut_ids[query_speaker_id]
                # query_cut_list.remove(cut_ids[index])
                if len(query_cut_list) == 0:
                    #no query utterance different from target utterance is found
                    return mixed_cuts
                query_id = random.choice(query_cut_list)
                query_cut = self.id2cuts[query_id]
                text = ""
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
        
    def get_bounded_segment(self, start_time, total_duration, min_duration=1.0, max_duration=10.0):
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
                cut_set.extend(self._create_non_existing_query_mixture(n_speakers=np.random.choice(np.arange(1, self.max_num_speakers+1))))           
        return CutSet.from_cuts(cut_set).shuffle()
