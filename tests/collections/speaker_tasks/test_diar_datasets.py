# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import os
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch.cuda

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.speaker_utils import get_vad_out_from_rttm_line, read_rttm_lines
from nemo.collections.common.parts.preprocessing.collections import EndtoEndDiarizationSpeechLabel


def is_rttm_length_too_long(rttm_file_path, wav_len_in_sec):
    """
    Check if the maximum RTTM duration exceeds the length of the provided audio file.

    Args:
        rttm_file_path (str): Path to the RTTM file.
        wav_len_in_sec (float): Length of the audio file in seconds.

    Returns:
        bool: True if the maximum RTTM duration is less than or equal to the length of the audio file, False otherwise.
    """
    rttm_lines = read_rttm_lines(rttm_file_path)
    max_rttm_sec = 0
    for line in rttm_lines:
        start, dur = get_vad_out_from_rttm_line(line)
        max_rttm_sec = max(max_rttm_sec, start + dur)
    return max_rttm_sec <= wav_len_in_sec


class TestAudioToSpeechE2ESpkDiarDataset:

    @staticmethod
    def _selection_dataset(
        feat_per_sec=10,
        start_guard_sec=0.2,
        min_first_speaker_sec=0.5,
        splice_silence_sec=0.2,
        max_speakers=4,
    ):
        dataset = object.__new__(AudioToSpeechE2ESpkDiarDataset)
        dataset.soft_label_thres = 0.5
        dataset.feat_per_sec = feat_per_sec
        dataset.subsegment_start_guard_sec = start_guard_sec
        dataset.subsegment_min_first_spk_sec = min_first_speaker_sec
        dataset.subsegment_splice_silence_sec = splice_silence_sec
        dataset.subsegment_nspk_bias = 1.0
        dataset.max_spks = max_speakers
        return dataset

    @staticmethod
    def _configured_dataset(tmp_path, **kwargs):
        session_len_sec = kwargs.pop("session_len_sec", 60.0)
        window_stride = kwargs.pop("window_stride", 0.1)
        manifest_path = tmp_path / "minimums_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": str(tmp_path / "audio.wav"),
                    "duration": 1.0,
                    "rttm_filepath": str(tmp_path / "audio.rttm"),
                }
            ),
            encoding="utf-8",
        )
        return AudioToSpeechE2ESpkDiarDataset(
            manifest_filepath=str(manifest_path),
            soft_label_thres=0.5,
            session_len_sec=session_len_sec,
            num_spks=2,
            featurizer=SimpleNamespace(sample_rate=100),
            fb_featurizer=SimpleNamespace(
                n_fft=16, hop_length=10, stft_pad_amount=16
            ),
            window_stride=window_stride,
            global_rank=0,
            soft_targets=False,
            device="cpu",
            validate_manifest_paths=False,
            **kwargs,
        )

    @staticmethod
    def _target(frames):
        return torch.tensor(
            [[("A" in frame), ("B" in frame), ("C" in frame)] for frame in frames],
            dtype=torch.bool,
        )

    def _is_eligible(self, dataset, frames, start=0, end=None):
        target = self._target(frames)
        end = len(frames) if end is None else end
        info = dataset._prepare_subsegment_activity(target)
        return dataset._source_chunk_eligibility(
            info, torch.tensor([start]), torch.tensor([end])
        ).item()

    @staticmethod
    def _eligibility_reference(dataset, activity_info, starts, ends):
        if starts.numel() == 0:
            return torch.zeros(0, dtype=torch.bool)
        starts = starts.long()
        ends = ends.long()
        num_frames, num_speakers = activity_info.activity.shape
        if num_speakers == 0:
            return torch.ones_like(starts, dtype=torch.bool)

        guard_frames = dataset._seconds_to_feature_frames(
            dataset.subsegment_start_guard_sec
        )
        minimum_frames = dataset._seconds_to_feature_frames(
            dataset.subsegment_min_first_spk_sec
        )
        candidate_silent = activity_info.active_count[starts] == 0
        effective_start = activity_info.next_active[starts]
        has_speech = effective_start < ends
        history_start = torch.clamp(starts - guard_frames, min=0)
        history = activity_info.prefix[starts] - activity_info.prefix[history_start]
        history_total = history.sum(dim=1)
        silence_history_safe = history_total == 0
        all_silence_eligible = candidate_silent & ~has_speech & silence_history_safe

        safe_effective_start = effective_start.clamp(max=max(num_frames - 1, 0))
        first_speaker = activity_info.first_speaker[safe_effective_start]
        exactly_one_speaker = activity_info.active_count[safe_effective_start] == 1
        first_speaker_history = history.gather(
            dim=1, index=first_speaker.unsqueeze(1)
        ).squeeze(1)
        history_safe = torch.where(
            candidate_silent,
            silence_history_safe,
            history_total - first_speaker_history == 0,
        )
        competitor = torch.minimum(
            activity_info.next_competitor[safe_effective_start], ends
        )
        accumulated = (
            activity_info.prefix[competitor, first_speaker]
            - activity_info.prefix[safe_effective_start, first_speaker]
        )
        condition_a = accumulated >= minimum_frames
        run_end = safe_effective_start + minimum_frames
        safe_run_end = run_end.clamp(max=num_frames)
        run_activity = (
            activity_info.prefix[safe_run_end, first_speaker]
            - activity_info.prefix[safe_effective_start, first_speaker]
        )
        condition_b = (run_end <= ends) & (run_activity >= minimum_frames)
        return all_silence_eligible | (
            has_speech
            & exactly_one_speaker
            & history_safe
            & (condition_a | condition_b)
        )

    @pytest.mark.unit
    def test_seconds_to_frames_uses_ceil_at_nonstandard_rate(self):
        dataset = self._selection_dataset(feat_per_sec=7)
        assert dataset._seconds_to_feature_frames(0.25) == 2
        assert dataset._seconds_to_feature_frames(0.50) == 4

    @pytest.mark.unit
    def test_optimized_eligibility_matches_full_history_reference(self):
        dataset = self._selection_dataset()
        generator = torch.Generator().manual_seed(3)
        target = torch.rand(200, 6, generator=generator) > 0.85
        info = dataset._prepare_subsegment_activity(target)
        starts = torch.randperm(190, generator=generator)[:120]
        lengths = torch.randint(1, 30, (120,), generator=generator)
        ends = torch.minimum(starts + lengths, torch.tensor(len(target)))

        expected = self._eligibility_reference(dataset, info, starts, ends)
        actual = dataset._source_chunk_eligibility(info, starts, ends)

        assert torch.equal(actual, expected)

    @pytest.mark.unit
    def test_optimized_eligibility_matches_reference_for_empty_candidates(self):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(torch.zeros(5, 2, dtype=torch.bool))
        starts = torch.zeros(0, dtype=torch.long)
        ends = torch.zeros(0, dtype=torch.long)
        assert torch.equal(
            dataset._source_chunk_eligibility(info, starts, ends),
            self._eligibility_reference(dataset, info, starts, ends),
        )

    @pytest.mark.unit
    def test_silence_candidate_without_future_speech_is_eligible(self):
        dataset = self._selection_dataset()
        assert self._is_eligible(dataset, ["-"] * 8, start=2)

    @pytest.mark.unit
    def test_silence_candidate_with_speech_in_guard_is_rejected(self):
        dataset = self._selection_dataset()
        assert not self._is_eligible(dataset, ["B", "-", "-", "-", "-"], start=2)

    @pytest.mark.unit
    def test_silence_candidate_followed_by_one_speaker_uses_evidence(self):
        dataset = self._selection_dataset()
        assert self._is_eligible(dataset, ["-", "-", "A", "A", "A", "A", "A"])
        assert not self._is_eligible(dataset, ["-", "-", "A", "A", "A", "A"])

    @pytest.mark.unit
    def test_silence_candidate_followed_by_simultaneous_speakers_is_rejected(self):
        dataset = self._selection_dataset()
        assert not self._is_eligible(dataset, ["-", "-", "AB", "AB", "AB", "AB", "AB"])

    @pytest.mark.unit
    def test_active_start_rejects_recent_different_speaker(self):
        dataset = self._selection_dataset()
        assert not self._is_eligible(
            dataset, ["B", "-", "A", "A", "A", "A", "A"], start=2
        )

    @pytest.mark.unit
    def test_active_start_allows_same_speaker_in_guard(self):
        dataset = self._selection_dataset()
        assert self._is_eligible(dataset, ["A", "A", "A", "A", "A", "A"], start=1)

    @pytest.mark.unit
    def test_missing_history_at_recording_start_is_safe(self):
        dataset = self._selection_dataset(start_guard_sec=1.0)
        assert self._is_eligible(dataset, ["A", "A", "A", "A", "A"])

    @pytest.mark.unit
    def test_multiple_speakers_at_effective_start_are_rejected(self):
        dataset = self._selection_dataset()
        assert not self._is_eligible(dataset, ["AB", "AB", "AB", "AB", "AB"])

    @pytest.mark.unit
    def test_first_speaker_accumulates_exact_threshold_across_silence(self):
        dataset = self._selection_dataset()
        frames = ["A", "A", "-", "-", "A", "A", "A", "B"]
        assert self._is_eligible(dataset, frames)

    @pytest.mark.unit
    def test_insufficient_accumulation_and_short_initial_segment_are_rejected(self):
        dataset = self._selection_dataset()
        assert not self._is_eligible(dataset, ["A", "A", "-", "A", "A", "B"])

    @pytest.mark.unit
    def test_first_speaker_activity_after_competitor_does_not_count(self):
        dataset = self._selection_dataset()
        frames = ["A", "A", "B", "-", "A", "A", "A", "A", "A"]
        assert not self._is_eligible(dataset, frames)

    @pytest.mark.unit
    def test_continuous_first_speaker_passes_while_competitor_joins(self):
        dataset = self._selection_dataset()
        assert self._is_eligible(dataset, ["A", "AB", "AB", "AB", "AB"])

    @pytest.mark.unit
    def test_inactive_frame_immediately_terminates_initial_run(self):
        dataset = self._selection_dataset()
        assert not self._is_eligible(dataset, ["A", "A", "-", "AB", "AB", "AB"])

    @pytest.mark.unit
    def test_exact_splice_silence_threshold_is_accepted(self):
        dataset = self._selection_dataset(splice_silence_sec=0.2)
        target = torch.zeros(8, 2, dtype=torch.bool)
        target[0, 0] = True
        target[3, 1] = True
        info = dataset._prepare_subsegment_activity(target)

        safe_end, safe_start = dataset._splice_silence_masks(info)

        assert safe_end[3]
        assert not safe_end[2]
        assert safe_start[1]
        assert not safe_start[2]

    @pytest.mark.unit
    def test_removed_subsegment_parameters_are_absent(self):
        parameters = inspect.signature(AudioToSpeechE2ESpkDiarDataset).parameters
        assert "subsegment_start_guard_frames" not in parameters
        assert "subsegment_min_first_spk_frames" not in parameters
        assert "subsegment_boundary_silence_frames" not in parameters
        assert "subsegment_min_len_sec" not in parameters
        assert "subsegment_min_chunk_len_sec" not in parameters
        assert "subsegment_margin_frames" not in parameters

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kwargs,expected_single,expected_two",
        [
            ({"subsegment_single_chunk_min_len_sec": 12.0}, 12.0, 10.0),
            ({"subsegment_two_chunk_min_len_sec": 8.0}, 15.0, 8.0),
            ({}, 15.0, 10.0),
            (
                {
                    "subsegment_single_chunk_min_len_sec": 11.0,
                    "subsegment_two_chunk_min_len_sec": 7.0,
                },
                11.0,
                7.0,
            ),
        ],
    )
    def test_chunk_minimum_configuration(
        self, tmp_path, kwargs, expected_single, expected_two
    ):
        dataset = self._configured_dataset(tmp_path, **kwargs)
        assert dataset.subsegment_single_chunk_min_len_sec == expected_single
        assert dataset.subsegment_two_chunk_min_len_sec == expected_two

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "parameter,value,error",
        [
            ("subsegment_single_chunk_min_len_sec", 0.0, ValueError),
            ("subsegment_single_chunk_min_len_sec", -1.0, ValueError),
            ("subsegment_single_chunk_min_len_sec", float("nan"), ValueError),
            ("subsegment_single_chunk_min_len_sec", float("inf"), ValueError),
            ("subsegment_single_chunk_min_len_sec", True, TypeError),
            ("subsegment_single_chunk_min_len_sec", "15", TypeError),
            ("subsegment_two_chunk_min_len_sec", 0.0, ValueError),
            ("subsegment_two_chunk_min_len_sec", -1.0, ValueError),
            ("subsegment_two_chunk_min_len_sec", float("nan"), ValueError),
            ("subsegment_two_chunk_min_len_sec", float("inf"), ValueError),
            ("subsegment_two_chunk_min_len_sec", True, TypeError),
            ("subsegment_two_chunk_min_len_sec", "10", TypeError),
        ],
    )
    def test_chunk_minimum_validation(self, tmp_path, parameter, value, error):
        with pytest.raises(error):
            self._configured_dataset(tmp_path, **{parameter: value})

    @pytest.mark.unit
    def test_single_minimum_cannot_exceed_active_session(self, tmp_path):
        with pytest.raises(ValueError, match="subsegment_single_chunk_min_len_sec"):
            self._configured_dataset(
                tmp_path,
                subsegment_mode=True,
                session_len_sec=10.0,
                subsegment_single_chunk_min_len_sec=11.0,
            )

    @pytest.mark.unit
    def test_two_chunk_minimum_constraint_only_applies_when_enabled(self, tmp_path):
        dataset = self._configured_dataset(
            tmp_path,
            subsegment_mode=True,
            session_len_sec=10.0,
            subsegment_single_chunk_min_len_sec=5.0,
            subsegment_two_chunk_min_len_sec=6.0,
            subsegment_two_chunks_rate=0.0,
        )
        assert dataset.subsegment_two_chunk_min_len_sec == 6.0

        with pytest.raises(ValueError, match="subsegment_two_chunk_min_len_sec"):
            self._configured_dataset(
                tmp_path,
                subsegment_mode=True,
                session_len_sec=10.0,
                subsegment_single_chunk_min_len_sec=5.0,
                subsegment_two_chunk_min_len_sec=6.0,
                subsegment_two_chunks_rate=0.5,
            )

    @pytest.mark.unit
    def test_frame_rounding_rejects_single_minimum_above_maximum(self, tmp_path):
        with pytest.raises(ValueError, match="26 frames"):
            self._configured_dataset(
                tmp_path,
                subsegment_mode=True,
                session_len_sec=0.255,
                window_stride=0.01,
                subsegment_single_chunk_min_len_sec=0.255,
            )

    @pytest.mark.unit
    def test_frame_rounding_rejects_two_minimums_above_maximum(self, tmp_path):
        with pytest.raises(ValueError, match="26 frames each"):
            self._configured_dataset(
                tmp_path,
                subsegment_mode=True,
                session_len_sec=0.51,
                window_stride=0.01,
                subsegment_single_chunk_min_len_sec=0.50,
                subsegment_two_chunk_min_len_sec=0.255,
                subsegment_two_chunks_rate=0.5,
            )

    @pytest.mark.unit
    def test_chunk_minimum_frame_conversion_uses_ceiling(self):
        dataset = self._selection_dataset(feat_per_sec=7)
        dataset.subsegment_single_chunk_min_len_sec = 0.50
        dataset.subsegment_two_chunk_min_len_sec = 0.255
        assert dataset._seconds_to_feature_frames(
            dataset.subsegment_single_chunk_min_len_sec
        ) == 4
        assert dataset._seconds_to_feature_frames(
            dataset.subsegment_two_chunk_min_len_sec
        ) == 2

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "parameter,value,error",
        [
            ("subsegment_start_guard_sec", -0.1, ValueError),
            ("subsegment_min_first_spk_sec", 0.0, ValueError),
            ("subsegment_splice_silence_sec", 0.0, ValueError),
            ("subsegment_start_guard_sec", True, TypeError),
            ("subsegment_min_first_spk_sec", "0.5", TypeError),
            ("subsegment_nspk_bias", 0.9, ValueError),
            ("subsegment_nspk_bias", float("nan"), ValueError),
            ("subsegment_nspk_bias", True, TypeError),
            ("subsegment_nspk_bias", "2.0", TypeError),
        ],
    )
    def test_subsegment_parameters_are_validated(
        self, tmp_path, parameter, value, error
    ):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": str(tmp_path / "audio.wav"),
                    "duration": 1.0,
                    "rttm_filepath": str(tmp_path / "audio.rttm"),
                }
            ),
            encoding="utf-8",
        )
        kwargs = {parameter: value}
        with pytest.raises(error):
            AudioToSpeechE2ESpkDiarDataset(
                manifest_filepath=str(manifest_path),
                soft_label_thres=0.5,
                session_len_sec=1.0,
                num_spks=2,
                featurizer=SimpleNamespace(sample_rate=100),
                fb_featurizer=SimpleNamespace(
                    n_fft=16, hop_length=10, stft_pad_amount=16
                ),
                window_stride=0.1,
                global_rank=0,
                soft_targets=False,
                device="cpu",
                validate_manifest_paths=False,
                **kwargs,
            )

    @pytest.mark.unit
    def test_single_chunk_rejects_windows_over_speaker_capacity(self):
        dataset = self._selection_dataset(min_first_speaker_sec=0.2, max_speakers=2)
        target = torch.zeros(30, 3, dtype=torch.bool)
        target[0:3, 0] = 1
        target[5:8, 1] = 1
        target[10:13, 2] = 1
        target[20:23, 0] = 1
        info = dataset._prepare_subsegment_activity(target)

        random.seed(0)
        bounds = dataset._sample_single_chunk_bounds(info, max_len=15, min_len=5)

        assert bounds is not None
        selected = torch.cat([target[start:end] for start, end in bounds])
        assert selected.any(dim=0).sum() <= dataset.max_spks

    @pytest.mark.unit
    def test_single_chunk_sampling_is_deterministic_with_fixed_seed(self):
        dataset = self._selection_dataset()
        target = self._target(["A"] * 8 + ["-"] * 4 + ["B"] * 8)
        info = dataset._prepare_subsegment_activity(target)

        random.seed(7)
        torch.manual_seed(7)
        first = dataset._sample_single_chunk_bounds(info, max_len=8, min_len=5)
        random.seed(7)
        torch.manual_seed(7)
        second = dataset._sample_single_chunk_bounds(info, max_len=8, min_len=5)

        assert first == second

    @pytest.mark.unit
    def test_short_clean_source_returns_full_available_length(self):
        dataset = self._selection_dataset(min_first_speaker_sec=0.3)
        target = self._target(["A"] * 4)
        info = dataset._prepare_subsegment_activity(target)

        bounds = dataset._sample_single_chunk_bounds(info, max_len=10, min_len=6)

        assert bounds == [(0, 4)]

    @pytest.mark.unit
    def test_source_equal_to_single_minimum_returns_full_source(self):
        dataset = self._selection_dataset()
        target = self._target(["A"] * 6)
        info = dataset._prepare_subsegment_activity(target)
        assert dataset._sample_single_chunk_bounds(info, max_len=10, min_len=6) == [
            (0, 6)
        ]

    @pytest.mark.unit
    def test_short_source_still_obeys_ats_and_capacity(self):
        dataset = self._selection_dataset(
            min_first_speaker_sec=0.2, max_speakers=1
        )
        overlap = dataset._prepare_subsegment_activity(self._target(["AB"] * 4))
        assert dataset._sample_single_chunk_bounds(overlap, max_len=10, min_len=6) is None

        over_capacity = dataset._prepare_subsegment_activity(
            self._target(["A", "A", "B", "B"])
        )
        assert (
            dataset._sample_single_chunk_bounds(
                over_capacity, max_len=10, min_len=6
            )
            is None
        )

    @pytest.mark.unit
    def test_empty_source_has_no_single_chunk_candidate(self):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(
            torch.zeros(0, 2, dtype=torch.bool)
        )
        assert dataset._sample_single_chunk_bounds(info, max_len=10, min_len=6) is None

    @pytest.mark.unit
    def test_failed_two_chunk_attempt_can_fall_back_to_short_source(self):
        dataset = self._selection_dataset(min_first_speaker_sec=0.3)
        target = self._target(["A"] * 4)
        info = dataset._prepare_subsegment_activity(target)

        assert dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3) is None
        assert dataset._sample_single_chunk_bounds(info, max_len=10, min_len=6) == [
            (0, 4)
        ]

    @pytest.mark.unit
    def test_single_chunk_bias_one_samples_uniformly(self, monkeypatch):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(
            torch.zeros(8, 2, dtype=torch.bool)
        )
        monkeypatch.setattr(random, "randrange", lambda size: size - 1)

        bounds = dataset._sample_single_chunk_bounds(info, max_len=5, min_len=3)

        assert bounds == [(5, 8)]

    @pytest.mark.unit
    def test_single_chunk_weighted_sampling_reuses_speaker_counts(self, monkeypatch):
        dataset = self._selection_dataset(max_speakers=2)
        dataset.subsegment_nspk_bias = 2.0
        info = dataset._prepare_subsegment_activity(
            torch.zeros(6, 3, dtype=torch.bool)
        )
        monkeypatch.setattr(
            dataset,
            "_source_chunk_eligibility",
            lambda activity_info, starts, ends: torch.ones_like(starts, dtype=torch.bool),
        )
        presence = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [False, False, True],
            ]
        )
        monkeypatch.setattr(
            dataset,
            "_chunk_speaker_presence",
            lambda activity_info, starts, ends: presence,
        )
        captured = {}

        def fake_multinomial(weights, num_samples):
            captured["weights"] = weights
            return torch.tensor([2])

        monkeypatch.setattr(torch, "multinomial", fake_multinomial)

        bounds = dataset._sample_single_chunk_bounds(info, max_len=3, min_len=3)

        assert torch.equal(captured["weights"], torch.tensor([2.0, 4.0, 2.0]))
        assert bounds == [(3, 6)]

    @pytest.mark.unit
    def test_two_chunk_selection_obeys_start_and_silence_rules(self):
        dataset = self._selection_dataset(
            min_first_speaker_sec=0.3,
            splice_silence_sec=0.2,
        )
        target = torch.zeros(80, 2, dtype=torch.bool)
        for start, end, speaker in (
            (0, 5, 0),
            (15, 20, 1),
            (30, 35, 0),
            (45, 50, 1),
            (60, 65, 0),
        ):
            target[start:end, speaker] = 1
        info = dataset._prepare_subsegment_activity(target)

        random.seed(0)
        torch.manual_seed(0)
        bounds = dataset._sample_two_chunk_bounds(info, total_len=24, min_chunk_len=8)

        assert bounds is not None
        (start1, end1), (start2, end2) = bounds
        starts = torch.tensor([start1, start2])
        ends = torch.tensor([end1, end2])
        safe_end, safe_start = dataset._splice_silence_masks(info)
        assert dataset._source_chunk_eligibility(info, starts, ends).all()
        assert safe_end[end1] and safe_start[start2]
        assert not target[start2].any()
        assert end1 - start1 >= 8 and end2 - start2 >= 8
        assert (end1 - start1) + (end2 - start2) == 24
        assert end2 <= start1 or start2 >= end1

    @pytest.mark.unit
    def test_two_chunk_selection_rejects_unions_over_speaker_capacity(self):
        dataset = self._selection_dataset(
            min_first_speaker_sec=0.3,
            splice_silence_sec=0.2,
            max_speakers=2,
        )
        target = torch.zeros(100, 3, dtype=torch.bool)
        for start, end, speaker in (
            (0, 5, 0),
            (20, 25, 1),
            (40, 45, 2),
            (60, 65, 0),
            (80, 85, 1),
        ):
            target[start:end, speaker] = 1
        info = dataset._prepare_subsegment_activity(target)

        random.seed(0)
        torch.manual_seed(0)
        bounds = dataset._sample_two_chunk_bounds(info, total_len=30, min_chunk_len=10)

        assert bounds is not None
        selected = torch.cat([target[start:end] for start, end in bounds])
        assert selected.any(dim=0).sum() <= dataset.max_spks

    @pytest.mark.unit
    def test_two_chunk_rejects_impossible_minimum_lengths(self):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(
            torch.zeros(20, 2, dtype=torch.bool)
        )
        assert dataset._sample_two_chunk_bounds(info, total_len=5, min_chunk_len=3) is None

    @pytest.mark.unit
    def test_two_chunk_prefers_full_remaining_length(self, monkeypatch):
        dataset = self._selection_dataset()
        target = torch.zeros(20, 2, dtype=torch.bool)
        info = dataset._prepare_subsegment_activity(target)
        monkeypatch.setattr(random, "randint", lambda low, high: 4)
        monkeypatch.setattr(random, "randrange", lambda size: 0)

        bounds = dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3)

        assert bounds is not None
        assert sum(end - start for start, end in bounds) == 10
        assert bounds[0][1] <= bounds[1][0]

    @pytest.mark.unit
    def test_two_chunk_shortened_fallback_works_when_source_is_shorter(self, monkeypatch):
        dataset = self._selection_dataset()
        target = torch.zeros(9, 2, dtype=torch.bool)
        info = dataset._prepare_subsegment_activity(target)
        monkeypatch.setattr(random, "randint", lambda low, high: 5)
        monkeypatch.setattr(random, "randrange", lambda size: 0)

        bounds = dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3)

        assert bounds == [(0, 5), (5, 9)]
        assert sum(end - start for start, end in bounds) == 9
        selected = torch.cat([target[start:end] for start, end in bounds])
        assert selected.shape[0] == 9

    @pytest.mark.unit
    def test_two_chunk_shortened_fallback_samples_among_longest_ties(self, monkeypatch):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(
            torch.zeros(12, 2, dtype=torch.bool)
        )
        choices = iter([4, 1])
        monkeypatch.setattr(random, "randint", lambda low, high: 4)
        monkeypatch.setattr(random, "randrange", lambda size: next(choices))

        bounds = dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3)

        assert bounds == [(4, 8), (8, 12)]

    @pytest.mark.unit
    def test_two_chunk_can_select_second_chunk_before_first(self, monkeypatch):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(
            torch.zeros(15, 2, dtype=torch.bool)
        )
        choices = iter([7, 0])
        monkeypatch.setattr(random, "randint", lambda low, high: 4)
        monkeypatch.setattr(random, "randrange", lambda size: next(choices))

        bounds = dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3)

        assert bounds == [(7, 11), (0, 6)]
        assert bounds[1][1] <= bounds[0][0]

    @pytest.mark.unit
    def test_two_chunk_one_shot_policy_does_not_retry_first_chunk(self, monkeypatch):
        dataset = self._selection_dataset()
        info = dataset._prepare_subsegment_activity(
            torch.zeros(9, 2, dtype=torch.bool)
        )
        choices = iter([2])
        randint_calls = []

        def fixed_length(low, high):
            randint_calls.append((low, high))
            return 5

        monkeypatch.setattr(random, "randint", fixed_length)
        monkeypatch.setattr(random, "randrange", lambda size: next(choices))

        bounds = dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3)

        assert bounds is None
        assert randint_calls == [(3, 7)]

    @pytest.mark.unit
    def test_two_chunk_bias_uses_first_and_union_speaker_counts(self, monkeypatch):
        dataset = self._selection_dataset(max_speakers=3)
        dataset.subsegment_nspk_bias = 2.0
        info = dataset._prepare_subsegment_activity(
            torch.zeros(15, 3, dtype=torch.bool)
        )
        presence_calls = []

        def fake_presence(activity_info, starts, ends):
            if not presence_calls:
                presence = torch.zeros(len(starts), 3, dtype=torch.bool)
                presence[:, 0] = True
                presence[1:, 1] = True
            else:
                presence = torch.zeros(len(starts), 3, dtype=torch.bool)
                presence[:, 0] = True
                if len(starts) > 1:
                    presence[1, 1] = True
            presence_calls.append(presence)
            return presence

        sampled_weights = []

        def fake_multinomial(weights, num_samples):
            sampled_weights.append(weights.clone())
            return torch.tensor([0 if len(sampled_weights) == 1 else 1])

        monkeypatch.setattr(dataset, "_chunk_speaker_presence", fake_presence)
        monkeypatch.setattr(random, "randint", lambda low, high: 4)
        monkeypatch.setattr(torch, "multinomial", fake_multinomial)

        bounds = dataset._sample_two_chunk_bounds(info, total_len=10, min_chunk_len=3)

        assert bounds is not None
        assert torch.equal(sampled_weights[0], torch.tensor([2.0] + [4.0] * 11))
        assert torch.equal(sampled_weights[1][:2], torch.tensor([2.0, 4.0]))

    @pytest.mark.unit
    def test_load_audio_chunks_reads_only_selected_ranges(self, monkeypatch):
        dataset = self._selection_dataset()
        dataset.feat_per_sec = 10

        class Featurizer:
            sample_rate = 100
            int_values = False

            @staticmethod
            def process_segment(segment):
                return torch.from_numpy(segment.samples)

        dataset.featurizer = Featurizer()
        calls = []

        def fake_from_file(audio_file, target_sr, int_values, offset, duration):
            calls.append((audio_file, offset, duration))
            value = float(len(calls))
            return AudioSegment(
                np.full(round(duration * target_sr), value, dtype=np.float32),
                target_sr,
            )

        monkeypatch.setattr(AudioSegment, "from_file", staticmethod(fake_from_file))

        audio = dataset._load_audio_chunks("audio.wav", offset=5.0, bounds=[(10, 20), (40, 55)])

        assert calls == [("audio.wav", 6.0, 1.0), ("audio.wav", 9.0, 1.5)]
        assert audio.shape == (250,)
        assert torch.all(audio[:100] == 1)
        assert torch.all(audio[100:] == 2)

    @pytest.mark.unit
    def test_create_subsegment_plans_before_loading_audio(self, tmp_path):
        rttm_path = tmp_path / "audio.rttm"
        rttm_path.write_text(
            "SPEAKER session 1 2.0 1.0 <NA> <NA> speaker_A <NA> <NA>\n",
            encoding="utf-8",
        )
        sample = SimpleNamespace(
            audio_file="audio.wav",
            rttm_file=str(rttm_path),
            duration=6.0,
        )
        dataset = self._selection_dataset()
        dataset.feat_per_sec = 10
        dataset.round_digits = 2
        dataset.session_len_sec = 2
        dataset.subsegment_single_chunk_min_len_sec = 1
        dataset.subsegment_two_chunk_min_len_sec = 0.5
        dataset.subsegment_two_chunks_rate = 0
        dataset.min_subsegment_duration = 0.03
        dataset.featurizer = SimpleNamespace(sample_rate=100)
        dataset._sample_two_chunk_bounds = lambda *args, **kwargs: pytest.fail(
            "Two-chunk selection must not run when its rate is zero"
        )

        def fake_single_chunk(activity_info, max_len, min_len):
            assert max_len == 20
            assert min_len == 10
            return [(20, 40)]

        dataset._sample_single_chunk_bounds = fake_single_chunk
        dataset._maybe_apply_opus_roundtrip = lambda audio: audio
        dataset.get_segment_timestamps = lambda duration, sample_rate: torch.tensor([20])
        dataset.get_frame_count_from_time_series_length = lambda length: 20
        dataset.get_soft_targets_seg = lambda feat_level_target, target_len: feat_level_target
        load_calls = []

        def fake_load(audio_file, offset, bounds):
            load_calls.append((audio_file, offset, bounds))
            return torch.zeros(200)

        dataset._load_audio_chunks = fake_load

        audio, audio_len, targets, target_len, speaker_names = dataset._create_subsegment(
            sample, offset=0
        )

        assert load_calls == [("audio.wav", 0, [(20, 40)])]
        assert audio.shape == (200,)
        assert audio_len == 200
        assert targets.shape == (20, dataset.max_spks)
        assert target_len == 20
        assert speaker_names[0] == "speaker_A"

    @pytest.mark.unit
    def test_create_subsegment_loads_full_short_source(self, tmp_path):
        rttm_path = tmp_path / "short.rttm"
        rttm_path.write_text(
            "SPEAKER session 1 0.0 0.4 <NA> <NA> speaker_A <NA> <NA>\n",
            encoding="utf-8",
        )
        sample = SimpleNamespace(
            audio_file="short.wav",
            rttm_file=str(rttm_path),
            duration=0.4,
        )
        dataset = self._selection_dataset(min_first_speaker_sec=0.2)
        dataset.round_digits = 2
        dataset.session_len_sec = 2.0
        dataset.subsegment_single_chunk_min_len_sec = 1.0
        dataset.subsegment_two_chunk_min_len_sec = 0.3
        dataset.subsegment_two_chunks_rate = 0
        dataset.min_subsegment_duration = 0.03
        dataset.featurizer = SimpleNamespace(sample_rate=100)
        dataset._maybe_apply_opus_roundtrip = lambda audio: audio
        dataset.get_segment_timestamps = lambda duration, sample_rate: torch.tensor([4])
        dataset.get_frame_count_from_time_series_length = lambda length: 4
        dataset.get_soft_targets_seg = lambda feat_level_target, target_len: feat_level_target
        load_calls = []

        def fake_load(audio_file, offset, bounds):
            load_calls.append((audio_file, offset, bounds))
            return torch.zeros(40)

        dataset._load_audio_chunks = fake_load

        audio, audio_len, targets, target_len, _ = dataset._create_subsegment(
            sample, offset=0
        )

        assert load_calls == [("short.wav", 0, [(0, 4)])]
        assert audio.shape == (40,)
        assert audio_len == 40
        assert targets.shape[0] == 4
        assert target_len == 4

    @pytest.mark.unit
    def test_rejected_subsegment_does_not_load_audio(self, tmp_path):
        rttm_path = tmp_path / "overlap.rttm"
        rttm_path.write_text(
            "\n".join(
                [
                    "SPEAKER session 1 0.0 1.0 <NA> <NA> speaker_A <NA> <NA>",
                    "SPEAKER session 1 0.0 1.0 <NA> <NA> speaker_B <NA> <NA>",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        sample = SimpleNamespace(
            audio_file="must-not-load.wav",
            rttm_file=str(rttm_path),
            duration=1.0,
        )
        dataset = self._selection_dataset()
        dataset.round_digits = 2
        dataset.session_len_sec = 0.5
        dataset.subsegment_single_chunk_min_len_sec = 0.5
        dataset.subsegment_two_chunk_min_len_sec = 0.2
        dataset.subsegment_two_chunks_rate = 0
        dataset.min_subsegment_duration = 0.03
        dataset.featurizer = SimpleNamespace(sample_rate=100)
        dataset._load_audio_chunks = lambda *args, **kwargs: pytest.fail(
            "Rejected candidates must not load audio"
        )

        audio, audio_len, targets, target_len, speaker_names = dataset._create_subsegment(
            sample, offset=0
        )

        assert audio.numel() == 0
        assert audio_len == 0
        assert targets.shape == (0, dataset.max_spks)
        assert target_len == 0
        assert speaker_names == [None] * dataset.max_spks

    @pytest.mark.unit
    def test_e2e_speaker_diar_dataset(self, test_data_dir):
        manifest_path = os.path.abspath(os.path.join(test_data_dir, 'asr/diarizer/lsm_val.json'))

        batch_size = 4
        num_samples = 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_dict_list = []
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            with open(manifest_path, 'r', encoding='utf-8') as mfile:
                for ix, line in enumerate(mfile):
                    if ix >= num_samples:
                        break

                    line = line.replace("tests/data/", test_data_dir + "/").replace("\n", "")
                    f.write(f"{line}\n")
                    data_dict = json.loads(line)
                    data_dict_list.append(data_dict)

            f.seek(0)
            featurizer = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)
            fb_featurizer = FilterbankFeatures(
                sample_rate=featurizer.sample_rate,
                n_window_size=int(0.025 * featurizer.sample_rate),
                n_window_stride=int(0.01 * featurizer.sample_rate),
                dither=False,
            )

            dataset = AudioToSpeechE2ESpkDiarDataset(
                manifest_filepath=f.name,
                soft_label_thres=0.5,
                session_len_sec=90,
                num_spks=4,
                featurizer=featurizer,
                window_stride=0.01,
                global_rank=0,
                soft_targets=False,
                device=device,
                fb_featurizer=fb_featurizer,
            )
            dataloader_instance = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.eesd_train_collate_fn,
                drop_last=False,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            assert len(dataloader_instance) == (num_samples / batch_size)  # Check if the number of batches is correct
            batch_counts = len(dataloader_instance)

            deviation_thres_rate = 0.01  # 1% deviation allowed
            for batch_index, batch in enumerate(dataloader_instance):
                audio_signals, audio_signal_len, targets, target_lens, speaker_names = batch
                if batch_index != batch_counts - 1:
                    assert audio_signals.shape[0] == batch_size, "Batch size does not match the expected value"
                assert len(speaker_names) == audio_signals.shape[0]
                for sample_index in range(audio_signals.shape[0]):
                    dataloader_audio_in_sec = audio_signal_len[sample_index].item()
                    data_dur_in_sec = abs(
                        data_dict_list[batch_size * batch_index + sample_index]['duration'] * featurizer.sample_rate
                        - dataloader_audio_in_sec
                    )
                    assert (
                        data_dur_in_sec <= deviation_thres_rate * dataloader_audio_in_sec
                    ), "Duration deviation exceeds 1%"
                assert not torch.isnan(audio_signals).any(), "audio_signals tensor contains NaN values"
                assert not torch.isnan(audio_signal_len).any(), "audio_signal_len tensor contains NaN values"
                assert not torch.isnan(targets).any(), "targets tensor contains NaN values"
                assert not torch.isnan(target_lens).any(), "target_lens tensor contains NaN values"

    @pytest.mark.unit
    def test_repeated_manifest_paths_are_checked_once(self, tmp_path, monkeypatch):
        audio_path = tmp_path / "audio.wav"
        rttm_path = tmp_path / "audio.rttm"
        manifest_path = tmp_path / "manifest.json"
        audio_path.touch()
        rttm_path.touch()

        entry = {
            "audio_filepath": str(audio_path),
            "duration": 1.0,
            "rttm_filepath": str(rttm_path),
        }
        manifest_path.write_text("\n".join(json.dumps(entry) for _ in range(3)), encoding="utf-8")

        checked_paths = []
        original_exists = os.path.exists

        def counting_exists(path):
            if path in (str(audio_path), str(rttm_path)):
                checked_paths.append(path)
            return original_exists(path)

        monkeypatch.setattr(os.path, "exists", counting_exists)

        collection = EndtoEndDiarizationSpeechLabel(manifests_files=str(manifest_path))

        assert len(collection) == 3
        assert checked_paths.count(str(audio_path)) == 1
        assert checked_paths.count(str(rttm_path)) == 1

    @pytest.mark.unit
    def test_manifest_path_validation_can_be_disabled(self, tmp_path, monkeypatch):
        audio_path = tmp_path / "missing.wav"
        rttm_path = tmp_path / "missing.rttm"
        manifest_path = tmp_path / "manifest.json"
        entry = {
            "audio_filepath": str(audio_path),
            "duration": 1.0,
            "rttm_filepath": str(rttm_path),
        }
        manifest_path.write_text(json.dumps(entry), encoding="utf-8")

        checked_paths = []
        original_exists = os.path.exists

        def counting_exists(path):
            if path in (str(audio_path), str(rttm_path)):
                checked_paths.append(path)
            return original_exists(path)

        monkeypatch.setattr(os.path, "exists", counting_exists)

        collection = EndtoEndDiarizationSpeechLabel(
            manifests_files=str(manifest_path),
            validate_manifest_paths=False,
        )

        assert len(collection) == 1
        assert checked_paths == []
