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
import os
import random
import tempfile

import pytest
import torch.cuda

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, WaveformFeaturizer
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
    def _selection_dataset(min_first_speaker_frames=5, boundary_silence_frames=2):
        dataset = object.__new__(AudioToSpeechE2ESpkDiarDataset)
        dataset.soft_label_thres = 0.5
        dataset.subsegment_min_first_spk_frames = min_first_speaker_frames
        dataset.subsegment_boundary_silence_frames = boundary_silence_frames
        dataset.subsegment_nspk_bias = 1.0
        return dataset

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "frames, expected",
        [
            (
                ["-", "-", "-", "-", "A", "A", "A", "-", "A", "A", "A", "A", "-", "B", "B"],
                True,
            ),
            (
                ["A", "A", "A", "-", "-", "A", "A", "AB", "B", "B", "B"],
                True,
            ),
            (
                ["-", "-", "-", "A", "A", "A", "A", "AB", "AB", "AB", "B", "B"],
                False,
            ),
        ],
    )
    def test_chunk_start_requires_established_first_speaker(self, frames, expected):
        dataset = self._selection_dataset()
        target = torch.tensor(
            [[float("A" in frame), float("B" in frame)] for frame in frames]
        )

        eligible = dataset._eligible_chunk_starts(target)

        assert eligible[0].item() is expected

    @pytest.mark.unit
    def test_chunk_start_vectorization_matches_reference(self):
        dataset = self._selection_dataset(min_first_speaker_frames=4)
        generator = torch.Generator().manual_seed(0)
        target = (torch.rand(50, 4, generator=generator) > 0.8).float()

        def reference(start):
            first_active = next(
                (frame for frame in range(start, len(target)) if target[frame].any()),
                None,
            )
            if first_active is None or target[first_active].sum() != 1:
                return False
            speaker = target[first_active].argmax()
            speaker_frames = 0
            for frame in range(first_active, len(target)):
                if target[frame].sum() - target[frame, speaker] > 0:
                    break
                speaker_frames += int(target[frame, speaker])
            return speaker_frames >= dataset.subsegment_min_first_spk_frames

        expected = torch.tensor([reference(start) for start in range(len(target))])
        assert torch.equal(dataset._eligible_chunk_starts(target), expected)

    @pytest.mark.unit
    def test_two_chunk_selection_obeys_start_and_silence_rules(self):
        dataset = self._selection_dataset(
            min_first_speaker_frames=3,
            boundary_silence_frames=2,
        )
        target = torch.zeros(80, 2)
        for start, end, speaker in (
            (0, 5, 0),
            (15, 20, 1),
            (30, 35, 0),
            (45, 50, 1),
            (60, 65, 0),
        ):
            target[start:end, speaker] = 1

        random.seed(0)
        torch.manual_seed(0)
        bounds = dataset._sample_two_chunk_bounds(target, total_len=24, min_chunk_len=8)

        assert bounds is not None
        (start1, end1), (start2, end2) = bounds
        eligible = dataset._eligible_chunk_starts(target)
        safe_boundary = dataset._silence_boundary_mask(target)
        assert eligible[start1] and eligible[start2]
        assert safe_boundary[end1] and safe_boundary[start2]
        assert end1 - start1 >= 8 and end2 - start2 >= 8
        assert (end1 - start1) + (end2 - start2) == 24
        assert end2 <= start1 or start2 >= end1

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
