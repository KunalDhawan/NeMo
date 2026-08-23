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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import SortformerEncLabelModel


@pytest.fixture()
def sortformer_model():

    model = {
        'sample_rate': 16000,
        'pil_weight': 0.5,
        'ats_weight': 0.5,
        'max_num_of_spks': 4,
        'async_streaming': False,
        'streaming_mode': False,
    }
    model_defaults = {
        'fc_d_model': 512,
        'tf_d_model': 192,
    }
    preprocessor = {
        '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'normalize': 'per_feature',
        'window_size': 0.025,
        'sample_rate': 16000,
        'window_stride': 0.01,
        'window': 'hann',
        'features': 80,
        'n_fft': 512,
        'frame_splicing': 1,
        'dither': 0.00001,
    }

    sortformer_modules = {
        '_target_': 'nemo.collections.asr.modules.sortformer_modules.SortformerModules',
        'num_spks': model['max_num_of_spks'],
        'dropout_rate': 0.5,
        'fc_d_model': model_defaults['fc_d_model'],
        'tf_d_model': model_defaults['tf_d_model'],
    }

    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
        'feat_in': preprocessor['features'],
        'feat_out': -1,
        'n_layers': 18,
        'd_model': model_defaults['fc_d_model'],
        'subsampling': 'dw_striding',
        'subsampling_factor': 8,
        'subsampling_conv_channels': 256,
        'causal_downsampling': False,
        'ff_expansion_factor': 4,
        'self_attention_model': 'rel_pos',
        'n_heads': 8,
        'att_context_size': [-1, -1],
        'att_context_style': 'regular',
        'xscaling': True,
        'untie_biases': True,
        'pos_emb_max_len': 5000,
        'conv_kernel_size': 9,
        'conv_norm_type': 'batch_norm',
        'conv_context_size': None,
        'dropout': 0.1,
        'dropout_pre_encoder': 0.1,
        'dropout_emb': 0.0,
        'dropout_att': 0.1,
        'stochastic_depth_drop_prob': 0.0,
        'stochastic_depth_mode': 'linear',
        'stochastic_depth_start_layer': 1,
    }

    transformer_encoder = {
        '_target_': 'nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder',
        'num_layers': 18,
        'hidden_size': model_defaults['tf_d_model'],
        'inner_size': 768,
        'num_attention_heads': 8,
        'attn_score_dropout': 0.5,
        'attn_layer_dropout': 0.5,
        'ffn_dropout': 0.5,
        'hidden_act': 'relu',
        'pre_ln': False,
        'pre_ln_final_layer_norm': True,
    }

    loss = {
        '_target_': 'nemo.collections.asr.losses.bce_loss.BCEWithLogitsLoss',
        'reduction': 'mean',
        'pos_weight': 1.0,
    }

    modelConfig = DictConfig(
        {
            'sample_rate': 16000,
            'pil_weight': 0.5,
            'ats_weight': 0.5,
            'activity_weight': 0.1,
            'presence_weight': 0.1,
            'presence_window_radius': 1,
            'presence_negative_margin': 0.4,
            'max_num_of_spks': 4,
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'transformer_encoder': DictConfig(transformer_encoder),
            'sortformer_modules': DictConfig(sortformer_modules),
            'preprocessor': DictConfig(preprocessor),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )
    model = SortformerEncLabelModel(cfg=modelConfig)
    return model


class TestSortformerEncLabelModelOffline:
    @pytest.mark.unit
    def test_constructor(self, sortformer_model):
        sortformer_model.streaming_mode = False
        sortformer_diar_model = sortformer_model.train()
        confdict = sortformer_diar_model.to_config_dict()
        instance2 = SortformerEncLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, SortformerEncLabelModel)

    @pytest.mark.unit
    def test_batch_active_speech_rms_uses_activity_and_lengths(self, sortformer_model):
        samples_per_target_frame = (
            sortformer_model.preprocessor.hop_length * sortformer_model.output_subsampling_factor
        )
        half_frame = samples_per_target_frame // 2
        max_audio_len = samples_per_target_frame + half_frame
        audio_signal = torch.full((2, max_audio_len), 100.0)
        audio_signal[0, :half_frame] = 2.0
        audio_signal[0, half_frame:samples_per_target_frame] = 4.0
        audio_signal[1, :half_frame] = 6.0
        audio_signal_length = torch.tensor([max_audio_len, half_frame])
        targets = torch.zeros(2, 2, 1)
        targets[0, 0, 0] = 1.0
        targets[1, 0, 0] = 1.0
        target_lens = torch.tensor([2, 1])

        active_rms, valid_audio_mask = sortformer_model._get_batch_active_speech_rms(
            audio_signal, audio_signal_length, targets, target_lens
        )

        assert torch.allclose(active_rms, torch.tensor([10.0**0.5, 6.0]))
        assert valid_audio_mask[0].all()
        assert valid_audio_mask[1, :half_frame].all()
        assert not valid_audio_mask[1, half_frame:].any()

    @pytest.mark.unit
    def test_batch_noise_augmentation_uses_individual_rms_and_original_batch(self, sortformer_model):
        sortformer_model.batch_noise_probability = 1.0
        sortformer_model.batch_noise_min_num_samples = 1
        sortformer_model.batch_noise_max_num_samples = 1
        sortformer_model.batch_noise_min_snr_db = 40.0
        sortformer_model.batch_noise_max_snr_db = 40.0

        audio_signal = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0],
            ]
        )
        original_audio = audio_signal.clone()
        audio_signal_length = torch.tensor([8, 6])
        targets = torch.zeros(2, 4, 1)
        targets[0, :4, 0] = 1.0
        targets[1, :3, 0] = 1.0
        target_lens = torch.tensor([4, 3])

        augmented = sortformer_model._apply_batch_noise_augmentation(
            audio_signal, audio_signal_length, targets, target_lens
        )

        expected = torch.tensor(
            [
                [2.02, 2.02, 2.02, 2.02, 2.02, 2.02, 2.0, 2.0],
                [4.04, 4.04, 4.04, 4.04, 4.04, 4.04, 0.0, 0.0],
            ]
        )
        assert torch.allclose(augmented, expected)
        assert torch.equal(audio_signal, original_audio)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (2, 1),  # Example 1
            (1, 2),  # Example 2
        ],
    )
    def test_forward_infer(self, sortformer_model, batch_size, sample_len):
        sortformer_model.streaming_mode = False
        sortformer_diar_model = sortformer_model.eval()
        confdict = sortformer_diar_model.to_config_dict()
        sampling_rate = confdict['preprocessor']['sample_rate']
        input_signal = torch.randn(size=(batch_size, sample_len * sampling_rate))
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        with torch.no_grad():
            # batch size 1
            preds_list = []
            for i in range(input_signal.size(0)):
                preds, _, _ = sortformer_diar_model.forward(
                    input_signal[i : i + 1], input_signal_length[i : i + 1]
                )
                preds_list.append(preds)
            preds_instance = torch.cat(preds_list, 0)

            # batch size 4
            preds_batch, _, _ = sortformer_diar_model.forward(input_signal, input_signal_length)
        assert preds_instance.shape == preds_batch.shape

        diff = torch.mean(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_activity_side_head_and_masked_loss(self, sortformer_model):
        model = sortformer_model.eval()
        hidden = torch.randn(2, 5, model.transformer_encoder.d_model)
        hidden_lens = torch.tensor([5, 3])

        with torch.no_grad():
            preds, speaker_logits, activity_logits = model.forward_infer(
                hidden,
                hidden_lens,
            )

        assert preds.shape == (2, 5, model.sortformer_modules.n_spk)
        assert speaker_logits.shape == preds.shape
        assert activity_logits.shape == (2, 5, 3)
        valid = torch.arange(preds.shape[1]).unsqueeze(0) < hidden_lens.unsqueeze(1)
        assert torch.allclose(preds[valid], torch.sigmoid(speaker_logits[valid]))
        assert torch.count_nonzero(preds[~valid]) == 0

        # Valid frames contain silence, single-speaker speech, and overlap. The final
        # padded frame is intentionally predicted incorrectly and must not affect loss.
        targets = torch.zeros(1, 4, model.sortformer_modules.n_spk)
        targets[0, 1, 0] = 1.0
        targets[0, 2, :2] = 1.0
        target_lens = torch.tensor([3])
        perfect_activity_logits = torch.tensor(
            [
                [
                    [10.0, -10.0, -10.0],
                    [-10.0, 10.0, -10.0],
                    [-10.0, -10.0, 10.0],
                    [-10.0, -10.0, 10.0],
                ]
            ],
            requires_grad=True,
        )

        activity_loss = model._activity_loss(
            perfect_activity_logits, targets, target_lens
        )
        assert activity_loss < 1e-3

        activity_loss.backward()
        assert perfect_activity_logits.grad is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("num_spks", [1, 2, 4, 8])
    def test_activity_logits_from_speaker_preds_matches_reference(self, sortformer_model, num_spks):
        model = sortformer_model.eval()
        generator = torch.Generator().manual_seed(0)
        base_preds = 0.05 + 0.9 * torch.rand(2, 5, num_spks, generator=generator)

        vectorized_preds = base_preds.clone().requires_grad_()
        vectorized_logits = model._activity_logits_from_speaker_preds(vectorized_preds)

        reference_preds = base_preds.clone().requires_grad_()
        prob_zero = torch.ones_like(reference_preds[..., 0])
        prob_one = torch.zeros_like(prob_zero)
        prob_overlap = torch.zeros_like(prob_zero)
        for speaker_idx in range(num_spks):
            speaker_prob = reference_preds[..., speaker_idx]
            prob_overlap = prob_overlap + prob_one * speaker_prob
            prob_one = prob_one * (1.0 - speaker_prob) + prob_zero * speaker_prob
            prob_zero = prob_zero * (1.0 - speaker_prob)
        reference_logits = torch.log(
            torch.stack((prob_zero, prob_one, prob_overlap), dim=-1) + model.eps
        )

        assert torch.allclose(vectorized_logits, reference_logits, atol=1e-5, rtol=1e-5)

        grad_output = torch.randn(vectorized_logits.shape, generator=generator)
        vectorized_grad = torch.autograd.grad(vectorized_logits, vectorized_preds, grad_output)[0]
        reference_grad = torch.autograd.grad(reference_logits, reference_preds, grad_output)[0]
        # Prefix/suffix products change float32 multiplication order relative to
        # the recurrence, and log gradients amplify small differences near zero.
        assert torch.allclose(vectorized_grad, reference_grad, atol=2e-4, rtol=1e-4)

    @pytest.mark.unit
    def test_activity_logits_from_speaker_preds_boundaries_and_permutation(self, sortformer_model):
        model = sortformer_model.eval()
        boundary_preds = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        expected_probs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        boundary_logits = model._activity_logits_from_speaker_preds(boundary_preds)
        boundary_probs = boundary_logits.exp() - model.eps
        assert torch.isfinite(boundary_logits).all()
        assert torch.allclose(boundary_probs, expected_probs, atol=1e-6)

        generator = torch.Generator().manual_seed(0)
        random_preds = torch.rand(2, 5, 8, generator=generator)
        activity_logits = model._activity_logits_from_speaker_preds(random_preds)
        permuted_logits = model._activity_logits_from_speaker_preds(
            random_preds[..., torch.randperm(8, generator=generator)]
        )
        activity_probs = activity_logits.exp() - model.eps
        assert torch.all(activity_probs >= -1e-6)
        assert torch.allclose(activity_probs.sum(dim=-1), torch.ones_like(activity_probs[..., 0]), atol=1e-6)
        assert torch.allclose(activity_logits, permuted_logits, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_activity_logits_from_speaker_preds_bfloat16(self, sortformer_model):
        model = sortformer_model.eval()
        preds = torch.rand(2, 5, 8, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0))
        logits = model._activity_logits_from_speaker_preds(preds)
        reference_logits = model._activity_logits_from_speaker_preds(preds.float())
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert torch.equal(logits, reference_logits)

    @pytest.mark.unit
    def test_activity_loss_from_speaker_preds(self, sortformer_model):
        model = sortformer_model.eval()
        model.activity_loss_mode = "speaker_preds"
        model.sortformer_modules.activity_head = None
        model._init_activity_auxiliary_heads()
        assert model.sortformer_modules.activity_head is None

        targets = torch.zeros(1, 4, model.sortformer_modules.n_spk)
        targets[0, 1, 0] = 1.0
        targets[0, 2, :2] = 1.0
        target_lens = torch.tensor([3])

        perfect_preds = torch.full_like(targets, 0.001)
        perfect_preds[0, 1, 0] = 0.999
        perfect_preds[0, 2, :2] = 0.999
        # Deliberately wrong prediction in padding must not affect the loss.
        perfect_preds[0, 3, :] = 0.999
        perfect_preds.requires_grad_()

        activity_logits = model._activity_logits_from_speaker_preds(perfect_preds)
        activity_loss = model._activity_loss(activity_logits, targets, target_lens)
        assert activity_loss < 0.02

        activity_loss.backward()
        assert torch.count_nonzero(perfect_preds.grad[:, :3]) > 0
        assert torch.count_nonzero(perfect_preds.grad[:, 3:]) == 0

    @pytest.mark.unit
    def test_self_ats_bce_uses_logits_and_ignores_padding(self, sortformer_model):
        model = sortformer_model.eval()
        model.self_ats_metric = "bce"
        model.self_ats_temperature = 0.5
        num_spks = model.sortformer_modules.n_spk

        sorted_logits = torch.full((1, 4, num_spks), -4.0)
        sorted_logits[0, 0, 0] = 4.0
        sorted_logits[0, 1, 1] = 4.0
        sorted_logits[0, 2, 2] = 4.0
        sorted_logits[0, 3, 3] = 10.0  # padding must not define an onset
        sorted_logits.requires_grad_()

        sorted_loss = model._self_ats_loss(sorted_logits, torch.tensor([3]))
        assert sorted_loss > 0  # BCE entropy floor
        sorted_loss.backward()
        assert torch.allclose(
            sorted_logits.grad,
            torch.zeros_like(sorted_logits.grad),
            atol=1e-6,
        )

        misordered_logits = sorted_logits.detach().clone()
        misordered_logits[0, 0, 0] = -4.0
        misordered_logits[0, 1, 0] = 4.0
        misordered_logits[0, 0, 1] = 4.0
        misordered_logits[0, 1, 1] = -4.0
        misordered_logits.requires_grad_()

        misordered_loss = model._self_ats_loss(
            misordered_logits, torch.tensor([3])
        )
        misordered_loss.backward()
        assert torch.count_nonzero(misordered_logits.grad[:, :3]) > 0
        assert torch.count_nonzero(misordered_logits.grad[:, 3:]) == 0

    @pytest.mark.unit
    def test_speaker_rank_loss_matches_margin_formula_and_masks_frames(self, sortformer_model):
        model = sortformer_model.eval()
        model.rank_margin = 0.6
        model.rank_collar_frames = 0

        speaker_logits = torch.tensor(
            [
                [
                    [1.0, 0.2, -0.4],
                    [0.1, 0.3, -0.2],
                    [-0.5, 0.4, 0.0],
                    [2.0, -1.0, 0.5],
                ],
                [
                    [-0.2, 0.7, 0.1],
                    [0.8, -0.3, 1.1],
                    [0.4, 0.2, -0.1],
                    [0.0, 0.0, 0.0],
                ],
            ],
            requires_grad=True,
        )
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 0, 0] = 1.0
        targets_pil[0, 1, :2] = 1.0  # overlap: ineligible
        targets_pil[0, 3, 0] = 1.0  # padding: ineligible
        targets_pil[1, 0, 1] = 1.0
        targets_pil[1, 1, 2] = 1.0
        targets_pil[1, 2, 0] = 1.0  # padding: ineligible
        target_lens = torch.tensor([3, 2])

        rank_loss = model._speaker_rank_loss(speaker_logits, targets_pil, target_lens)

        expected_terms = []
        for batch_index, frame_index, target_index in ((0, 0, 0), (1, 0, 1), (1, 1, 2)):
            frame_logits = speaker_logits[batch_index, frame_index]
            competitor_mask = torch.arange(frame_logits.numel()) != target_index
            expected_terms.append(
                torch.log1p(
                    torch.exp(
                        frame_logits[competitor_mask]
                        - frame_logits[target_index]
                        + model.rank_margin
                    ).sum()
                )
            )
        expected_loss = torch.stack(expected_terms).mean()

        assert torch.allclose(rank_loss, expected_loss)

        rank_loss.backward()
        assert speaker_logits.grad[0, 0, 0] < 0
        assert torch.all(speaker_logits.grad[0, 0, 1:] > 0)
        assert torch.count_nonzero(speaker_logits.grad[0, 1:]) == 0
        assert torch.count_nonzero(speaker_logits.grad[1, :2]) > 0
        assert torch.count_nonzero(speaker_logits.grad[1, 2:]) == 0

    @pytest.mark.unit
    def test_speaker_rank_loss_excludes_transition_collar(self, sortformer_model):
        model = sortformer_model.eval()
        model.rank_margin = 0.0
        model.rank_collar_frames = 1

        speaker_logits = torch.zeros(1, 7, 3, requires_grad=True)
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, :2, 0] = 1.0
        targets_pil[0, 2:5, 1] = 1.0
        targets_pil[0, 5:, 0] = 1.0

        rank_loss = model._speaker_rank_loss(speaker_logits, targets_pil, torch.tensor([7]))

        assert torch.allclose(rank_loss, torch.log(torch.tensor(3.0)))

        rank_loss.backward()
        assert torch.count_nonzero(speaker_logits.grad[0, [0, 3, 6]]) > 0
        assert torch.count_nonzero(speaker_logits.grad[0, [1, 2, 4, 5]]) == 0

    @pytest.mark.unit
    def test_speaker_rank_loss_uses_global_ddp_denominator(
        self, sortformer_model, monkeypatch
    ):
        model = sortformer_model.eval()
        model.rank_margin = 0.0
        model.rank_collar_frames = 0

        speaker_logits = torch.tensor(
            [[[1.0, 0.0, -1.0], [-0.5, 0.5, 0.0]]],
            requires_grad=True,
        )
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 0, 0] = 1.0
        targets_pil[0, 1, 1] = 1.0

        def fake_all_reduce(counts, op):
            del op
            counts.add_(3.0)

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        rank_loss = model._speaker_rank_loss(speaker_logits, targets_pil, torch.tensor([2]))

        local_terms = torch.stack(
            (
                torch.log1p(torch.exp(speaker_logits[0, 0, 1:] - speaker_logits[0, 0, 0]).sum()),
                torch.log1p(
                    torch.exp(
                        speaker_logits[0, 1, [0, 2]] - speaker_logits[0, 1, 1]
                    ).sum()
                ),
            )
        )
        assert torch.allclose(rank_loss, 2.0 * local_terms.sum() / 5.0)

    @pytest.mark.unit
    def test_speech_bce_loss_matches_per_frame_slot_mean_and_masks_silence(
        self, sortformer_model
    ):
        model = sortformer_model.eval()
        model.speech_bce_collar_frames = 0

        speaker_logits = torch.tensor(
            [
                [
                    [1.0, 0.2, -0.4],
                    [0.1, 0.3, -0.2],
                    [-0.5, 0.4, 0.0],
                    [2.0, -1.0, 0.5],
                ],
                [
                    [-0.2, 0.7, 0.1],
                    [0.8, -0.3, 1.1],
                    [0.4, 0.2, -0.1],
                    [0.0, 0.0, 0.0],
                ],
            ],
            requires_grad=True,
        )
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 0] = torch.tensor([0.8, 0.2, 0.0])
        targets_pil[0, 1, :2] = torch.tensor([0.8, 0.7])  # soft overlap is eligible
        targets_pil[0, 3, 0] = 1.0  # padding is ineligible
        targets_pil[1, 1] = torch.tensor([0.1, 0.0, 0.9])
        targets_pil[1, 2, 0] = 1.0  # padding is ineligible
        target_lens = torch.tensor([3, 2])

        speech_bce_loss = model._speech_bce_loss(
            speaker_logits, targets_pil, target_lens
        )
        selected_logits = torch.stack(
            (speaker_logits[0, 0], speaker_logits[0, 1], speaker_logits[1, 1])
        )
        selected_targets = torch.stack(
            (targets_pil[0, 0], targets_pil[0, 1], targets_pil[1, 1])
        )
        expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            selected_logits,
            selected_targets,
            reduction="none",
        ).mean(dim=-1).mean()

        assert torch.allclose(speech_bce_loss, expected_loss)

        speech_bce_loss.backward()
        assert torch.count_nonzero(speaker_logits.grad[0, :2]) > 0
        assert torch.count_nonzero(speaker_logits.grad[0, 2:]) == 0
        assert torch.count_nonzero(speaker_logits.grad[1, 0]) == 0
        assert torch.count_nonzero(speaker_logits.grad[1, 1]) > 0
        assert torch.count_nonzero(speaker_logits.grad[1, 2:]) == 0

    @pytest.mark.unit
    def test_speech_bce_loss_excludes_transition_collar(self, sortformer_model):
        model = sortformer_model.eval()
        model.speech_bce_collar_frames = 1

        speaker_logits = torch.zeros(1, 7, 3, requires_grad=True)
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, :2, 0] = 1.0
        targets_pil[0, 2:5, 1] = 1.0
        targets_pil[0, 5:, 0] = 1.0

        speech_bce_loss = model._speech_bce_loss(
            speaker_logits, targets_pil, torch.tensor([7])
        )

        assert torch.allclose(speech_bce_loss, torch.log(torch.tensor(2.0)))

        speech_bce_loss.backward()
        assert torch.count_nonzero(speaker_logits.grad[0, [0, 3, 6]]) > 0
        assert torch.count_nonzero(speaker_logits.grad[0, [1, 2, 4, 5]]) == 0

    @pytest.mark.unit
    def test_speech_bce_loss_uses_global_ddp_frame_count(
        self, sortformer_model, monkeypatch
    ):
        model = sortformer_model.eval()
        model.speech_bce_collar_frames = 0

        speaker_logits = torch.tensor(
            [[[1.0, 0.0, -1.0], [-0.5, 0.5, 0.0]]],
            requires_grad=True,
        )
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 0, 0] = 1.0
        targets_pil[0, 1, 1] = 1.0

        def fake_all_reduce(count, op):
            del op
            count.add_(3.0)

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        speech_bce_loss = model._speech_bce_loss(
            speaker_logits, targets_pil, torch.tensor([2])
        )
        local_frame_losses = torch.nn.functional.binary_cross_entropy_with_logits(
            speaker_logits,
            targets_pil,
            reduction="none",
        ).mean(dim=-1)

        assert torch.allclose(
            speech_bce_loss,
            2.0 * local_frame_losses.sum() / 5.0,
        )

    @pytest.mark.unit
    def test_interior_focal_loss_uses_strict_regions_and_joint_class_mean(
        self, sortformer_model
    ):
        model = sortformer_model.eval()
        model.interior_focal_gamma = 2.0
        model.interior_focal_positive_radius = 1
        model.interior_focal_negative_radius = 1

        speaker_logits = torch.linspace(-2.0, 2.0, 22).reshape(1, 11, 2)
        speaker_logits.requires_grad_()
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 3:8, 0] = 1.0
        targets_pil[0, 5, 1] = 1.0
        target_lens = torch.tensor([11])

        positive_eligible = torch.zeros_like(targets_pil, dtype=torch.bool)
        positive_eligible[0, 4:7, 0] = True
        negative_eligible = torch.zeros_like(targets_pil, dtype=torch.bool)
        negative_eligible[0, [1, 9], 0] = True
        negative_eligible[0, 1:4, 1] = True
        negative_eligible[0, 7:10, 1] = True

        focal_loss = model._interior_focal_loss(
            speaker_logits, targets_pil, target_lens
        )
        positive_loss = (
            torch.sigmoid(-speaker_logits).pow(model.interior_focal_gamma)
            * torch.nn.functional.softplus(-speaker_logits)
        )
        negative_loss = (
            torch.sigmoid(speaker_logits).pow(model.interior_focal_gamma)
            * torch.nn.functional.softplus(speaker_logits)
        )
        expected = (
            positive_loss.masked_select(positive_eligible).sum()
            + negative_loss.masked_select(negative_eligible).sum()
        ) / (positive_eligible.sum() + negative_eligible.sum())

        assert torch.allclose(focal_loss, expected)

        focal_loss.backward()
        eligible = positive_eligible | negative_eligible
        assert torch.count_nonzero(speaker_logits.grad[eligible]) > 0
        assert torch.count_nonzero(speaker_logits.grad[~eligible]) == 0

    @pytest.mark.unit
    def test_interior_focal_loss_uses_global_ddp_eligible_count(
        self, sortformer_model, monkeypatch
    ):
        model = sortformer_model.eval()
        model.interior_focal_gamma = 2.0
        model.interior_focal_positive_radius = 1
        model.interior_focal_negative_radius = 1

        speaker_logits = torch.linspace(-2.0, 2.0, 22).reshape(1, 11, 2)
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 3:8, 0] = 1.0
        targets_pil[0, 5, 1] = 1.0
        target_lens = torch.tensor([11])

        positive_eligible = torch.zeros_like(targets_pil, dtype=torch.bool)
        positive_eligible[0, 4:7, 0] = True
        negative_eligible = torch.zeros_like(targets_pil, dtype=torch.bool)
        negative_eligible[0, [1, 9], 0] = True
        negative_eligible[0, 1:4, 1] = True
        negative_eligible[0, 7:10, 1] = True
        local_count = positive_eligible.sum() + negative_eligible.sum()
        local_sum = (
            (
                torch.sigmoid(-speaker_logits).pow(model.interior_focal_gamma)
                * torch.nn.functional.softplus(-speaker_logits)
            )
            .masked_select(positive_eligible)
            .sum()
            + (
                torch.sigmoid(speaker_logits).pow(model.interior_focal_gamma)
                * torch.nn.functional.softplus(speaker_logits)
            )
            .masked_select(negative_eligible)
            .sum()
        )

        def fake_all_reduce(count, op):
            del op
            count.add_(7.0)

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        loss = model._interior_focal_loss(
            speaker_logits, targets_pil, target_lens
        )
        assert torch.allclose(loss, 2.0 * local_sum / (local_count + 7.0))

    @pytest.mark.unit
    def test_purity_focal_loss_uses_soft_purity_weights(
        self, sortformer_model
    ):
        model = sortformer_model.eval()
        model.purity_focal_gamma = 2.0
        model.purity_focal_power = 2.0
        model.purity_focal_positive_radius = 1
        model.purity_focal_negative_radius = 1

        speaker_logits = torch.linspace(-2.0, 2.0, 22).reshape(1, 11, 2)
        speaker_logits.requires_grad_()
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 3:8, 0] = 1.0
        targets_pil[0, 5, 1] = 0.6
        target_lens = torch.tensor([10])
        valid = (
            torch.arange(speaker_logits.shape[1]).unsqueeze(0)
            < target_lens.unsqueeze(1)
        ).unsqueeze(-1)
        positive_support = targets_pil * valid
        negative_support = (1.0 - targets_pil) * valid
        positive_purity = torch.nn.functional.avg_pool1d(
            positive_support.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        ).transpose(1, 2)
        negative_purity = torch.nn.functional.avg_pool1d(
            negative_support.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        ).transpose(1, 2)
        positive_weight = (
            positive_support
            * positive_purity.pow(model.purity_focal_power)
        )
        negative_weight = (
            negative_support
            * negative_purity.pow(model.purity_focal_power)
        )

        focal_loss = model._purity_focal_loss(
            speaker_logits, targets_pil, target_lens
        )
        positive_loss = (
            torch.sigmoid(-speaker_logits).pow(model.purity_focal_gamma)
            * torch.nn.functional.softplus(-speaker_logits)
        )
        negative_loss = (
            torch.sigmoid(speaker_logits).pow(model.purity_focal_gamma)
            * torch.nn.functional.softplus(speaker_logits)
        )
        expected = (
            positive_weight * positive_loss
            + negative_weight * negative_loss
        ).sum() / (positive_weight.sum() + negative_weight.sum())

        assert torch.allclose(focal_loss, expected)

        focal_loss.backward()
        assert torch.count_nonzero(speaker_logits.grad[:, :10]) > 0
        assert torch.count_nonzero(speaker_logits.grad[:, 10:]) == 0

        model.purity_focal_gamma = 0.0
        model.purity_focal_power = 0.0
        gamma_zero_loss = model._purity_focal_loss(
            speaker_logits.detach(), targets_pil, target_lens
        )
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            speaker_logits.detach()[:, :10],
            targets_pil[:, :10],
        )
        assert torch.allclose(gamma_zero_loss, expected_bce)

    @pytest.mark.unit
    def test_purity_focal_loss_uses_global_ddp_weight_sum(
        self, sortformer_model, monkeypatch
    ):
        model = sortformer_model.eval()
        model.purity_focal_gamma = 2.0
        model.purity_focal_power = 2.0
        model.purity_focal_positive_radius = 1
        model.purity_focal_negative_radius = 1

        speaker_logits = torch.linspace(-2.0, 2.0, 22).reshape(1, 11, 2)
        targets_pil = torch.zeros_like(speaker_logits)
        targets_pil[0, 3:8, 0] = 1.0
        targets_pil[0, 5, 1] = 0.6
        target_lens = torch.tensor([10])
        valid = (
            torch.arange(speaker_logits.shape[1]).unsqueeze(0)
            < target_lens.unsqueeze(1)
        ).unsqueeze(-1)
        positive_support = targets_pil * valid
        negative_support = (1.0 - targets_pil) * valid
        positive_purity = torch.nn.functional.avg_pool1d(
            positive_support.transpose(1, 2),
            3,
            stride=1,
            padding=1,
            count_include_pad=True,
        ).transpose(1, 2)
        negative_purity = torch.nn.functional.avg_pool1d(
            negative_support.transpose(1, 2),
            3,
            stride=1,
            padding=1,
            count_include_pad=True,
        ).transpose(1, 2)
        positive_weight = (
            positive_support
            * positive_purity.pow(model.purity_focal_power)
        )
        negative_weight = (
            negative_support
            * negative_purity.pow(model.purity_focal_power)
        )
        positive_loss = (
            torch.sigmoid(-speaker_logits).pow(model.purity_focal_gamma)
            * torch.nn.functional.softplus(-speaker_logits)
        )
        negative_loss = (
            torch.sigmoid(speaker_logits).pow(model.purity_focal_gamma)
            * torch.nn.functional.softplus(speaker_logits)
        )
        local_sum = (
            positive_weight * positive_loss
            + negative_weight * negative_loss
        ).sum()
        local_weight_sum = positive_weight.sum() + negative_weight.sum()

        def fake_all_reduce(weight_sum, op):
            del op
            weight_sum.add_(7.0)

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        loss = model._purity_focal_loss(
            speaker_logits, targets_pil, target_lens
        )
        assert torch.allclose(loss, 2.0 * local_sum / (local_weight_sum + 7.0))

    @pytest.mark.unit
    def test_pil_aligned_windowed_presence_loss(self, sortformer_model):
        model = sortformer_model.eval()
        num_spks = model.sortformer_modules.n_spk
        targets_pil = torch.zeros(1, 6, num_spks)
        targets_pil[0, 2, 1] = 1.0
        target_lens = torch.tensor([5])

        aligned_preds = torch.full_like(targets_pil, 0.001)
        aligned_preds[0, 2, 1] = 0.999
        aligned_preds.requires_grad_()
        aligned_loss = model._presence_loss(aligned_preds, targets_pil, target_lens)

        misaligned_preds = torch.full_like(targets_pil, 0.001)
        misaligned_preds[0, 2, 0] = 0.999
        misaligned_loss = model._presence_loss(misaligned_preds, targets_pil, target_lens)

        padded_fake_preds = aligned_preds.detach().clone()
        padded_fake_preds[0, 5, 0] = 0.999
        padded_fake_loss = model._presence_loss(padded_fake_preds, targets_pil, target_lens)

        assert aligned_loss < 0.01
        assert misaligned_loss > aligned_loss + 1.0
        assert torch.allclose(padded_fake_loss, aligned_loss)

        absent_targets = torch.zeros_like(targets_pil)
        safe_hedge_preds = torch.full_like(targets_pil, model.presence_negative_margin - 0.01)
        safe_hedge_preds.requires_grad_()
        safe_hedge_loss = model._presence_loss(safe_hedge_preds, absent_targets, target_lens)
        unsafe_hedge_preds = safe_hedge_preds.detach().clone()
        unsafe_hedge_preds[0, 2, 0] = model.presence_negative_margin + 0.2
        unsafe_hedge_loss = model._presence_loss(unsafe_hedge_preds, absent_targets, target_lens)

        assert torch.equal(safe_hedge_loss, torch.tensor(0.0))
        assert unsafe_hedge_loss > safe_hedge_loss

        aligned_loss.backward()
        assert aligned_preds.grad is not None
        safe_hedge_loss.backward()
        assert torch.count_nonzero(safe_hedge_preds.grad) == 0

    @pytest.mark.unit
    def test_pil_aligned_dice_loss_ignores_target_absent_speakers(self, sortformer_model):
        sortformer_model.dice_min_target_frames = 1
        targets_pil = torch.zeros(1, 4, 2)
        targets_pil[0, :2, 0] = 1.0
        target_lens = torch.tensor([3])

        perfect_preds = targets_pil.clone()
        # Predictions in padding must not affect the loss.
        perfect_preds[0, 3, :] = 1.0
        perfect_loss = sortformer_model._dice_loss(perfect_preds, targets_pil, target_lens)

        phantom_preds = perfect_preds.clone()
        phantom_preds[0, 2, 1] = 1.0
        phantom_preds.requires_grad_()
        phantom_loss = sortformer_model._dice_loss(phantom_preds, targets_pil, target_lens)

        active_false_alarm_preds = perfect_preds.clone()
        active_false_alarm_preds[0, 2, 0] = 1.0
        active_false_alarm_loss = sortformer_model._dice_loss(
            active_false_alarm_preds, targets_pil, target_lens
        )

        assert torch.equal(perfect_loss, torch.tensor(0.0))
        assert torch.equal(phantom_loss, perfect_loss)
        assert active_false_alarm_loss > perfect_loss

        phantom_loss.backward()
        assert phantom_preds.grad[0, 2, 1] == 0
        assert torch.count_nonzero(phantom_preds.grad[0, 3]) == 0

    @pytest.mark.unit
    def test_pil_aligned_dice_loss_requires_minimum_target_duration(self, sortformer_model):
        sortformer_model.dice_min_target_frames = 2
        targets_pil = torch.zeros(1, 4, 2)
        targets_pil[0, :2, 0] = 1.0
        targets_pil[0, 0, 1] = 1.0
        target_lens = torch.tensor([3])

        short_speaker_miss = targets_pil.clone()
        short_speaker_miss[0, 0, 1] = 0.0
        short_speaker_loss = sortformer_model._dice_loss(
            short_speaker_miss, targets_pil, target_lens
        )

        eligible_speaker_miss = targets_pil.clone()
        eligible_speaker_miss[0, 1, 0] = 0.0
        eligible_speaker_loss = sortformer_model._dice_loss(
            eligible_speaker_miss, targets_pil, target_lens
        )

        assert torch.equal(short_speaker_loss, torch.tensor(0.0))
        assert eligible_speaker_loss > short_speaker_loss

    @pytest.mark.unit
    def test_dice_loss_weights_speakers_by_target_duration(self, sortformer_model):
        model = sortformer_model
        model.dice_min_target_frames = 1
        model.dice_duration_gamma = 0.5

        targets_pil = torch.zeros(1, 4, 2)
        targets_pil[0, 0, 0] = 1.0
        targets_pil[0, :, 1] = 1.0
        preds = torch.zeros_like(targets_pil, requires_grad=True)
        target_lens = torch.tensor([4])

        weighted_loss = model._dice_loss(preds, targets_pil, target_lens)
        # With zero predictions and smooth=1, losses are 1/2 and 4/5.
        # Gamma 0.5 gives target-mass weights sqrt(1)=1 and sqrt(4)=2.
        expected_weighted = (
            torch.tensor(0.5) + 2.0 * torch.tensor(0.8)
        ) / 3.0
        assert torch.allclose(weighted_loss, expected_weighted)

        model.dice_duration_gamma = 0.0
        equal_speaker_loss = model._dice_loss(preds, targets_pil, target_lens)
        assert torch.allclose(
            equal_speaker_loss,
            (torch.tensor(0.5) + torch.tensor(0.8)) / 2.0,
        )

        weighted_loss.backward()
        assert torch.count_nonzero(preds.grad) > 0

    @pytest.mark.unit
    def test_dice_loss_uses_global_ddp_weight_sum(self, sortformer_model, monkeypatch):
        model = sortformer_model
        model.dice_min_target_frames = 1
        model.dice_duration_gamma = 0.5

        targets_pil = torch.zeros(1, 4, 2)
        targets_pil[0, 0, 0] = 1.0
        targets_pil[0, :, 1] = 1.0
        preds = torch.zeros_like(targets_pil)
        target_lens = torch.tensor([4])

        def fake_all_reduce(weight_sum, op):
            del op
            weight_sum.add_(5.0)

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        loss = model._dice_loss(preds, targets_pil, target_lens)
        local_weighted_sum = torch.tensor(0.5) + 2.0 * torch.tensor(0.8)
        assert torch.allclose(loss, 2.0 * local_weighted_sum / 8.0)

    @pytest.mark.unit
    def test_phantom_loss_penalizes_only_high_confidence_empty_channels(self, sortformer_model):
        num_spks = sortformer_model.sortformer_modules.n_spk
        targets_pil = torch.zeros(1, 4, num_spks)
        targets_pil[0, 0, 0] = 1.0
        target_lens = torch.tensor([3])

        one_phantom_probs = torch.full_like(targets_pil, 0.1)
        # Activity on a non-empty channel is outside the scope of this loss.
        one_phantom_probs[0, :3, 0] = 0.9
        # Two selected frames on one empty channel.
        one_phantom_probs[0, 1, 1] = 0.5
        one_phantom_probs[0, 2, 1] = 0.75
        # An empty channel below the threshold and activity in padding are ignored.
        one_phantom_probs[0, 1, 2] = sortformer_model.phantom_threshold - 0.01
        one_phantom_probs[0, 3, 3] = 0.9
        one_phantom_logits = torch.logit(one_phantom_probs).requires_grad_()

        one_phantom_loss = sortformer_model._phantom_loss(
            one_phantom_logits, targets_pil, target_lens
        )
        expected_channel_loss = (-torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.25))) / 2
        assert torch.allclose(one_phantom_loss, expected_channel_loss / num_spks)

        # The fixed speaker-slot denominator makes a second identical phantom
        # channel add an equal amount to the loss.
        two_phantom_probs = one_phantom_probs.clone()
        two_phantom_probs[0, 1, 2] = 0.5
        two_phantom_probs[0, 2, 2] = 0.75
        two_phantom_logits = torch.logit(two_phantom_probs)
        two_phantom_loss = sortformer_model._phantom_loss(
            two_phantom_logits, targets_pil, target_lens
        )
        assert torch.allclose(two_phantom_loss, 2 * one_phantom_loss)

        one_phantom_loss.backward()
        assert torch.all(one_phantom_logits.grad[0, 1:3, 1] > 0)
        assert torch.count_nonzero(one_phantom_logits.grad[0, :, 0]) == 0
        assert one_phantom_logits.grad[0, 1, 2] == 0
        assert torch.count_nonzero(one_phantom_logits.grad[0, 3]) == 0

        sortformer_model.phantom_logmeanexp = True
        sortformer_model.phantom_logmeanexp_temperature = 0.5
        logmeanexp_loss = sortformer_model._phantom_loss(
            one_phantom_logits.detach(), targets_pil, target_lens
        )
        selected_losses = torch.stack(
            (-torch.log(torch.tensor(0.5)), -torch.log(torch.tensor(0.25)))
        )
        expected_logmeanexp = sortformer_model.phantom_logmeanexp_temperature * (
            torch.logsumexp(
                selected_losses / sortformer_model.phantom_logmeanexp_temperature,
                dim=0,
            )
            - torch.log(torch.tensor(2.0))
        )
        assert torch.allclose(logmeanexp_loss, expected_logmeanexp / num_spks)
        assert logmeanexp_loss > one_phantom_loss

    @pytest.mark.unit
    def test_phantom_entry_loss_focuses_first_offending_chunk(self, sortformer_model):
        model = sortformer_model.eval()
        model.sortformer_modules.chunk_len = 3
        model.upsample_factor = 1
        assert model.phantom_entry_threshold == 0.5
        assert model.phantom_threshold == 0.25

        num_spks = model.sortformer_modules.n_spk
        probs = torch.full((1, 10, num_spks), 0.1)
        targets_pil = torch.zeros_like(probs)
        targets_pil[0, 0, 0] = 1.0
        target_lens = torch.tensor([9])

        # Non-empty target channels are outside the scope of this loss.
        probs[0, :9, 0] = 0.9
        # Channel 1 first crosses threshold in chunk 1 (frames 3-5).
        probs[0, 3, 1] = 0.3  # included by the lower aggregation threshold
        probs[0, 4, 1] = 0.6
        probs[0, 5, 1] = 0.8
        probs[0, 6:8, 1] = 0.9  # later persistence is ignored
        # Channel 2 stays below entry threshold until chunk 2.
        probs[0, 1, 2] = 0.4
        probs[0, 6, 2] = 0.3  # included once chunk 2 is chosen
        probs[0, 7, 2] = 0.55
        probs[0, 8, 2] = 0.7
        # A lower-threshold value alone does not create an entry.
        probs[0, 2, 3] = 0.4
        # Activity in padding must not create an entry either.
        probs[0, 9, 3] = 0.9
        speaker_logits = torch.logit(probs).requires_grad_()

        entry_loss = model._phantom_entry_loss(
            speaker_logits, targets_pil, target_lens
        )
        channel_one_loss = (
            -torch.log(torch.tensor(0.7))
            - torch.log(torch.tensor(0.4))
            - torch.log(torch.tensor(0.2))
        ) / 3
        channel_two_loss = (
            -torch.log(torch.tensor(0.7))
            - torch.log(torch.tensor(0.45))
            - torch.log(torch.tensor(0.3))
        ) / 3
        assert torch.allclose(
            entry_loss,
            (channel_one_loss + channel_two_loss) / num_spks,
        )

        entry_loss.backward()
        selected = torch.zeros_like(speaker_logits, dtype=torch.bool)
        selected[0, 3:6, 1] = True
        selected[0, 6:9, 2] = True
        assert torch.all(speaker_logits.grad[selected] > 0)
        assert torch.count_nonzero(speaker_logits.grad[~selected]) == 0

        model.phantom_logmeanexp = True
        model.phantom_logmeanexp_temperature = 0.5
        logmeanexp_loss = model._phantom_entry_loss(
            speaker_logits.detach(), targets_pil, target_lens
        )

        def logmeanexp_channel_loss(channel_probs):
            frame_losses = -torch.log1p(-torch.tensor(channel_probs))
            temperature = model.phantom_logmeanexp_temperature
            return temperature * (
                torch.logsumexp(frame_losses / temperature, dim=0)
                - torch.log(torch.tensor(float(len(channel_probs))))
            )

        expected_logmeanexp = (
            logmeanexp_channel_loss([0.3, 0.6, 0.8])
            + logmeanexp_channel_loss([0.3, 0.55, 0.7])
        ) / num_spks
        assert torch.allclose(logmeanexp_loss, expected_logmeanexp)
        assert logmeanexp_loss > entry_loss

    @pytest.mark.unit
    def test_prearrival_loss_penalizes_only_activity_safely_before_first_onset(
        self, sortformer_model
    ):
        model = sortformer_model.eval()
        model.prearrival_threshold = 0.25
        model.prearrival_grace_frames = 1

        num_spks = model.sortformer_modules.n_spk
        probs = torch.full((1, 7, num_spks), 0.1)
        targets_pil = torch.zeros_like(probs)
        target_lens = torch.tensor([6])
        targets_pil[0, 2:, 0] = 1.0  # speaker 0 first arrives at frame 2
        targets_pil[0, 5, 1] = 1.0  # speaker 1 first arrives at frame 5
        targets_pil[0, 6, 3] = 1.0  # padding: speaker 3 is target-empty in valid frames

        # Speaker 0: only frame 0 is earlier than the one-frame grace window.
        probs[0, 0, 0] = 0.5
        probs[0, 1, 0] = 0.9
        probs[0, 3, 0] = 0.9
        # Speaker 1: two selected frames are averaged within the channel.
        probs[0, 1, 1] = 0.5
        probs[0, 2, 1] = 0.75
        probs[0, 4, 1] = 0.9
        # Never-active speakers remain eligible throughout valid frames.
        probs[0, 4, 2] = 0.6
        probs[0, 5, 3] = 0.4
        # Padding and sub-threshold predictions are ignored.
        probs[0, 6, 3] = 0.9
        speaker_logits = torch.logit(probs).requires_grad_()

        prearrival_loss = model._prearrival_loss(
            speaker_logits, targets_pil, target_lens
        )
        channel_losses = torch.stack(
            (
                -torch.log(torch.tensor(0.5)),
                (-torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.25))) / 2,
                -torch.log(torch.tensor(0.4)),
                -torch.log(torch.tensor(0.6)),
            )
        )
        assert torch.allclose(prearrival_loss, channel_losses.sum() / num_spks)

        prearrival_loss.backward()
        selected = torch.zeros_like(speaker_logits, dtype=torch.bool)
        selected[0, 0, 0] = True
        selected[0, 1:3, 1] = True
        selected[0, 4, 2] = True
        selected[0, 5, 3] = True
        assert torch.all(speaker_logits.grad[selected] > 0)
        assert torch.count_nonzero(speaker_logits.grad[~selected]) == 0

        model.prearrival_logmeanexp = True
        model.prearrival_logmeanexp_temperature = 0.5
        logmeanexp_loss = model._prearrival_loss(
            speaker_logits.detach(), targets_pil, target_lens
        )
        speaker_one_losses = torch.stack(
            (-torch.log(torch.tensor(0.5)), -torch.log(torch.tensor(0.25)))
        )
        speaker_one_logmeanexp = model.prearrival_logmeanexp_temperature * (
            torch.logsumexp(
                speaker_one_losses / model.prearrival_logmeanexp_temperature,
                dim=0,
            )
            - torch.log(torch.tensor(2.0))
        )
        expected_logmeanexp = torch.stack(
            (
                -torch.log(torch.tensor(0.5)),
                speaker_one_logmeanexp,
                -torch.log(torch.tensor(0.4)),
                -torch.log(torch.tensor(0.6)),
            )
        ).sum() / num_spks
        assert torch.allclose(logmeanexp_loss, expected_logmeanexp)
        assert logmeanexp_loss > prearrival_loss

    @pytest.mark.unit
    def test_global_speaker_count_metrics(self, sortformer_model):
        num_spks = sortformer_model.sortformer_modules.n_spk
        preds = torch.zeros(3, 4, num_spks)
        targets = torch.zeros_like(preds)
        target_lens = torch.tensor([4, 3, 2])

        # Sample 0: predict three speakers when two are present (absolute error 1).
        targets[0, 0, :2] = 1.0
        preds[0, 0, :3] = 0.9
        # Sample 1: count is correct even though the active channel differs.
        targets[1, 1, 2] = 1.0
        preds[1, 1, 0] = 0.9
        # Sample 2: activity only in padding must not count as a speaker.
        preds[2, 2, 3] = 0.9

        count_mae, count_accuracy = sortformer_model._speaker_count_metrics(
            preds, targets, target_lens
        )

        assert torch.allclose(count_mae, torch.tensor(1.0 / 3.0))
        assert torch.allclose(count_accuracy, torch.tensor(2.0 / 3.0))

    @pytest.mark.unit
    def test_multi_validation_epoch_end_includes_auxiliary_losses(self, sortformer_model):
        scalar_keys = [
            'val_loss',
            'val_ats_loss',
            'val_pil_loss',
            'val_rank_loss',
            'val_speech_bce_loss',
            'val_interior_focal_loss',
            'val_purity_focal_loss',
            'val_pairwise_ats_loss',
            'val_self_ats_loss',
            'val_spkcount_loss',
            'val_activity_loss',
            'val_presence_loss',
            'val_dice_loss',
            'val_phantom_loss',
            'val_phantom_entry_loss',
            'val_prearrival_loss',
            'val_f1_acc',
            'val_precision',
            'val_recall',
            'val_f1_acc_ats',
            'val_speaker_count_mae',
            'val_speaker_count_accuracy',
        ]
        outputs = [
            {key: torch.tensor(1.0) for key in scalar_keys},
            {key: torch.tensor(3.0) for key in scalar_keys},
        ]

        metrics = sortformer_model.multi_validation_epoch_end(outputs)['log']

        assert torch.equal(metrics['val_rank_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_speech_bce_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_interior_focal_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_purity_focal_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_activity_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_presence_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_dice_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_phantom_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_phantom_entry_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_prearrival_loss'], torch.tensor(2.0))


class TestSortformerEncLabelModelStreaming:
    @pytest.mark.unit
    def test_constructor(self, sortformer_model):
        sortformer_model.streaming_mode = True
        sortformer_diar_model = sortformer_model.train()
        confdict = sortformer_diar_model.to_config_dict()
        instance2 = SortformerEncLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, SortformerEncLabelModel)

    @pytest.mark.unit
    def test_chunk_replaced_targets_preserve_returning_recipient_speakers(self, sortformer_model):
        recipient_targets = torch.zeros(10, 4)
        recipient_targets[:4, 0] = 1.0
        recipient_targets[8:, 0] = 1.0
        recipient_targets[8:, 2] = 1.0
        donor_targets = torch.zeros(10, 4)
        donor_targets[:2, 0] = 1.0
        donor_targets[2:4, 1] = 0.4

        replaced_targets = sortformer_model._build_chunk_replaced_targets(
            recipient_targets=recipient_targets,
            donor_targets=donor_targets,
            recipient_speaker_names=["A1", None, "A2", None],
            donor_speaker_names=["B1", "B2", None, None],
            recipient_target_len=10,
            donor_target_len=10,
            target_frames_per_chunk=2,
            destination_chunks=(2, 3),
        )

        assert torch.equal(replaced_targets[:4], recipient_targets[:4])
        assert torch.all(replaced_targets[4:6, 1] == 1.0)
        assert torch.all(replaced_targets[6:8, 3] == 0.4)
        assert torch.all(replaced_targets[8:, 0] == 1.0)
        assert torch.all(replaced_targets[8:, 2] == 1.0)

        with pytest.raises(RuntimeError, match="overlaps the retained recipient"):
            sortformer_model._build_chunk_replaced_targets(
                recipient_targets=recipient_targets,
                donor_targets=donor_targets,
                recipient_speaker_names=["A1", None, "A2", None],
                donor_speaker_names=["A1", "B2", None, None],
                recipient_target_len=10,
                donor_target_len=10,
                target_frames_per_chunk=2,
                destination_chunks=(2, 3),
            )

    @pytest.mark.unit
    def test_batch_chunk_replacement_uses_original_donor_prefixes(self, sortformer_model):
        model = sortformer_model
        model.batch_chunk_replace_probability = 1.0
        model.batch_chunk_replace_min_num_chunks = 3
        model.batch_chunk_replace_max_num_chunks = 3
        model.batch_chunk_replace_num_preserved_chunks = 2
        model.sortformer_modules.chunk_len = 2

        feature_frames_per_chunk = (
            model.sortformer_modules.chunk_len * model.sortformer_modules.subsampling_factor
        )
        processed_signal = torch.zeros(2, 1, 5 * feature_frames_per_chunk)
        for chunk_index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
            start = chunk_index * feature_frames_per_chunk
            processed_signal[0, :, start : start + feature_frames_per_chunk] = value
        for chunk_index, value in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
            start = chunk_index * feature_frames_per_chunk
            processed_signal[1, :, start : start + feature_frames_per_chunk] = value
        original_signal = processed_signal.clone()
        processed_signal_length = torch.tensor([processed_signal.shape[2]] * 2)

        targets = torch.zeros(2, 10, 4)
        targets[0, :, 0] = 1.0
        targets[1, :2, 0] = 1.0
        targets[1, 2:4, 1] = 1.0
        original_targets = targets.clone()
        target_lens = torch.tensor([10, 10])
        speaker_names = [
            ["A1", None, None, None],
            ["B1", "B2", None, None],
        ]

        augmented_signal, augmented_targets, replacement_rate = (
            model._apply_batch_chunk_replace_augmentation(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                targets=targets,
                target_lens=target_lens,
                speaker_names=speaker_names,
            )
        )

        assert torch.equal(
            augmented_signal[0, :, : 2 * feature_frames_per_chunk],
            original_signal[0, :, : 2 * feature_frames_per_chunk],
        )
        assert torch.all(
            augmented_signal[0, :, 2 * feature_frames_per_chunk : 3 * feature_frames_per_chunk]
            == 10.0
        )
        assert torch.all(
            augmented_signal[0, :, 3 * feature_frames_per_chunk : 4 * feature_frames_per_chunk]
            == 20.0
        )
        assert torch.all(augmented_signal[0, :, 4 * feature_frames_per_chunk :] == 30.0)
        assert torch.all(
            augmented_signal[1, :, 2 * feature_frames_per_chunk : 3 * feature_frames_per_chunk]
            == 1.0
        )
        assert torch.all(
            augmented_signal[1, :, 3 * feature_frames_per_chunk : 4 * feature_frames_per_chunk]
            == 2.0
        )
        # The third source chunk must still be the donor's original value, not its prior replacement.
        assert torch.all(augmented_signal[1, :, 4 * feature_frames_per_chunk :] == 3.0)
        assert torch.equal(augmented_targets[0, :4, 0], original_targets[0, :4, 0])
        assert torch.equal(processed_signal, original_signal)
        assert torch.equal(targets, original_targets)
        assert replacement_rate == 1.0

    @pytest.mark.unit
    def test_batch_chunk_replacement_uses_only_valid_partial_donor_data(self, sortformer_model):
        model = sortformer_model
        model.batch_chunk_replace_probability = 1.0
        model.batch_chunk_replace_min_num_chunks = 1
        model.batch_chunk_replace_max_num_chunks = 1
        model.batch_chunk_replace_num_preserved_chunks = 2
        model.sortformer_modules.chunk_len = 2

        feature_frames_per_chunk = (
            model.sortformer_modules.chunk_len * model.sortformer_modules.subsampling_factor
        )
        partial_chunk_length = feature_frames_per_chunk // 2
        recipient_length = 2 * feature_frames_per_chunk + partial_chunk_length
        processed_signal = torch.full((2, 1, recipient_length), -99.0)
        processed_signal[0, :, :recipient_length] = 1.0
        processed_signal[1, :, :partial_chunk_length] = 7.0
        processed_signal_length = torch.tensor([recipient_length, partial_chunk_length])

        targets = torch.zeros(2, 5, 4)
        targets[0, :, 0] = 1.0
        targets[1, 0, 0] = 1.0
        target_lens = torch.tensor([5, 1])
        speaker_names = [["A1", None, None, None], ["B1", None, None, None]]

        augmented_signal, augmented_targets, replacement_rate = (
            model._apply_batch_chunk_replace_augmentation(
                processed_signal,
                processed_signal_length,
                targets,
                target_lens,
                speaker_names,
            )
        )
        destination_start = 2 * feature_frames_per_chunk
        assert torch.all(augmented_signal[0, :, destination_start:recipient_length] == 7.0)
        assert augmented_targets[0, 4, 1] == 1.0
        assert replacement_rate == 0.5

        # One fewer valid donor feature makes the plan ineligible; padded values are never copied.
        too_short_lengths = torch.tensor([recipient_length, partial_chunk_length - 1])
        skipped_signal, skipped_targets, skipped_rate = model._apply_batch_chunk_replace_augmentation(
            processed_signal,
            too_short_lengths,
            targets,
            target_lens,
            speaker_names,
        )
        assert torch.equal(skipped_signal, processed_signal)
        assert torch.equal(skipped_targets, targets)
        assert skipped_rate == 0.0

        # Nonzero target padding must not make a donor with insufficient target length eligible.
        padded_targets = targets.clone()
        padded_targets[1, 0, 0] = 1.0
        too_short_target_lens = torch.tensor([5, 0])
        padded_signal, padded_result_targets, padded_rate = (
            model._apply_batch_chunk_replace_augmentation(
                processed_signal,
                processed_signal_length,
                padded_targets,
                too_short_target_lens,
                speaker_names,
            )
        )
        assert torch.equal(padded_signal, processed_signal)
        assert torch.equal(padded_result_targets, padded_targets)
        assert padded_rate == 0.0

    @pytest.mark.unit
    def test_batch_chunk_replacement_enforces_speaker_constraints(self, sortformer_model, monkeypatch):
        model = sortformer_model
        model.batch_chunk_replace_probability = 1.0
        model.batch_chunk_replace_min_num_chunks = 2
        model.batch_chunk_replace_max_num_chunks = 2
        model.batch_chunk_replace_num_preserved_chunks = 2
        model.sortformer_modules.chunk_len = 2

        feature_frames_per_chunk = (
            model.sortformer_modules.chunk_len * model.sortformer_modules.subsampling_factor
        )
        processed_signal = torch.randn(2, 1, 4 * feature_frames_per_chunk)
        processed_signal_length = torch.tensor([processed_signal.shape[2]] * 2)
        target_lens = torch.tensor([8, 8])

        # Three recipient speakers plus two disjoint donor speakers would exceed the four-speaker limit.
        targets = torch.zeros(2, 8, 4)
        targets[0, :4, :3] = 1.0
        targets[1, :4, :2] = 1.0
        speaker_names = [
            ["A1", "A2", "A3", None],
            ["B1", "B2", None, None],
        ]
        capped_signal, capped_targets, capped_rate = model._apply_batch_chunk_replace_augmentation(
            processed_signal, processed_signal_length, targets, target_lens, speaker_names
        )
        assert torch.equal(capped_signal, processed_signal)
        assert torch.equal(capped_targets, targets)
        assert capped_rate == 0.0

        # A shared ID anywhere in the full sessions makes the pair ineligible, even when
        # that donor speaker is outside the prefix that would be copied.
        overlapping_targets = torch.zeros(2, 8, 4)
        overlapping_targets[0, :4, 0] = 1.0
        overlapping_targets[1, :2, 0] = 1.0
        overlapping_targets[1, 2:4, 1] = 1.0
        overlapping_targets[1, 6:8, 2] = 1.0
        overlapping_names = [
            ["shared", None, None, None],
            ["B1", "B2", "shared", None],
        ]
        monkeypatch.setattr(
            model,
            "_select_chunk_replacement_plan",
            lambda *args, **kwargs: pytest.fail("incompatible recipient should not run plan search"),
        )
        overlap_signal, overlap_targets, overlap_rate = model._apply_batch_chunk_replace_augmentation(
            processed_signal,
            processed_signal_length,
            overlapping_targets,
            target_lens,
            overlapping_names,
        )
        assert torch.equal(overlap_signal, processed_signal)
        assert torch.equal(overlap_targets, overlapping_targets)
        assert overlap_rate == 0.0

    @pytest.mark.unit
    def test_chunk_replacement_requires_net_speaker_count_increase(self, sortformer_model):
        model = sortformer_model
        model.batch_chunk_replace_min_num_chunks = 1
        model.batch_chunk_replace_max_num_chunks = 1
        model.batch_chunk_replace_num_preserved_chunks = 2
        model.sortformer_modules.chunk_len = 2
        feature_frames_per_chunk = (
            model.sortformer_modules.chunk_len * model.sortformer_modules.subsampling_factor
        )
        processed_signal = torch.zeros(2, 1, 3 * feature_frames_per_chunk)
        processed_signal_length = torch.tensor([processed_signal.shape[2]] * 2)
        target_lens = torch.tensor([6, 6])

        targets = torch.zeros(2, 6, 4)
        targets[0, :4, 0] = 1.0
        targets[0, 4:, 1] = 1.0
        targets[1, :2, 0] = 1.0
        speaker_names = [
            ["A1", "A2", None, None],
            ["B1", None, None, None],
        ]
        metadata = model._build_chunk_replacement_metadata(
            processed_signal,
            processed_signal_length,
            targets,
            target_lens,
            speaker_names,
        )
        assert model._select_chunk_replacement_plan(0, metadata) is None

        # Removing A2 but inserting B1+B2 gives a genuine net increase from two to three.
        targets[1, :2, 1] = 1.0
        speaker_names[1][1] = "B2"
        metadata = model._build_chunk_replacement_metadata(
            processed_signal,
            processed_signal_length,
            targets,
            target_lens,
            speaker_names,
        )
        assert model._select_chunk_replacement_plan(0, metadata) is not None

        # Capacity is limited by the target tensor as well as max_num_of_spks.
        narrow_targets = torch.zeros(2, 6, 3)
        narrow_targets[0, :4, 0] = 1.0
        narrow_targets[1, :2, :] = 1.0
        narrow_names = [["A1", None, None], ["B1", "B2", "B3"]]
        narrow_metadata = model._build_chunk_replacement_metadata(
            processed_signal,
            processed_signal_length,
            narrow_targets,
            target_lens,
            narrow_names,
        )
        assert model._select_chunk_replacement_plan(0, narrow_metadata) is None

    @pytest.mark.unit
    def test_chunk_replacement_probability_is_per_recipient(self, sortformer_model, monkeypatch):
        model = sortformer_model
        model.batch_chunk_replace_probability = 0.25
        metadata = SimpleNamespace(
            feature_lengths=torch.ones(4, dtype=torch.long),
            compatible_pairs=~torch.eye(4, dtype=torch.bool),
        )
        draws = iter((0.10, 0.30, 0.24, 0.90))
        monkeypatch.setattr(
            "nemo.collections.asr.models.sortformer_diar_models.random.random",
            lambda: next(draws),
        )
        monkeypatch.setattr(
            model,
            "_select_chunk_replacement_plan",
            lambda recipient_index, metadata: recipient_index,
        )

        assert model._sample_chunk_replacement_plans(metadata) == [0, 2]

    @pytest.mark.unit
    def test_training_step_uses_chunk_augmented_targets(self, sortformer_model, monkeypatch):
        model = sortformer_model.train()
        model.streaming_mode = True
        model.batch_noise_probability = 0.0
        model.batch_chunk_replace_probability = 1.0
        model.sortformer_modules.activity_head = None

        audio_signal = torch.zeros(2, 8)
        audio_signal_length = torch.tensor([8, 8])
        targets = torch.zeros(2, 4, 4)
        augmented_targets = torch.ones_like(targets)
        target_lens = torch.tensor([4, 4])
        speaker_names = [["A", None, None, None], ["B", None, None, None]]
        processed_signal = torch.zeros(2, 80, 8)
        processed_signal_length = torch.tensor([8, 8])
        captured = {}

        monkeypatch.setattr(
            model,
            "process_signal",
            lambda audio_signal, audio_signal_length: (processed_signal, processed_signal_length),
        )

        def fake_chunk_augmentation(**kwargs):
            assert kwargs["speaker_names"] == speaker_names
            return kwargs["processed_signal"], augmented_targets, torch.tensor(0.5)

        monkeypatch.setattr(model, "_apply_batch_chunk_replace_augmentation", fake_chunk_augmentation)

        def fake_forward_streaming(
            processed_signal,
            processed_signal_length,
        ):
            del processed_signal, processed_signal_length
            preds = torch.zeros_like(targets)
            return preds, torch.zeros_like(preds), None

        monkeypatch.setattr(model, "forward_streaming", fake_forward_streaming)

        def fake_train_evaluations(
            preds, augmented, lengths, activity_logits=None, speaker_logits=None
        ):
            del preds, lengths, activity_logits, speaker_logits
            captured["targets"] = augmented
            return {"loss": torch.tensor(1.0)}

        monkeypatch.setattr(model, "_get_aux_train_evaluations", fake_train_evaluations)
        monkeypatch.setattr(model, "_reset_train_metrics", lambda: None)
        monkeypatch.setattr(model, "log_dict", lambda metrics, **kwargs: captured.update(metrics))

        output = model.training_step(
            (audio_signal, audio_signal_length, targets, target_lens, speaker_names),
            batch_idx=0,
        )

        assert torch.equal(captured["targets"], augmented_targets)
        assert captured["batch_chunk_replace_rate"] == 0.5
        assert output["loss"] == 1.0

    @pytest.mark.unit
    def test_forward_with_activity_logits(self, sortformer_model):
        model = sortformer_model.eval()
        model.streaming_mode = True
        sample_len = model.preprocessor._cfg.sample_rate
        input_signal = torch.randn(1, sample_len)
        input_signal_length = torch.tensor([sample_len])

        with torch.no_grad():
            preds, speaker_logits, activity_logits = model.forward(
                input_signal,
                input_signal_length,
            )

        assert speaker_logits.shape == preds.shape
        assert activity_logits.shape[:2] == preds.shape[:2]
        assert torch.allclose(preds, torch.sigmoid(speaker_logits), atol=1e-6)
        assert activity_logits.shape[2] == 3

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (2, 1),  # Example 1
            (1, 2),  # Example 2
        ],
    )
    def test_forward_infer(self, sortformer_model, batch_size, sample_len):
        sortformer_model.streaming_mode = True
        sortformer_diar_model = sortformer_model.eval()
        confdict = sortformer_diar_model.to_config_dict()
        sampling_rate = confdict['preprocessor']['sample_rate']
        input_signal = torch.randn(size=(batch_size, sample_len * sampling_rate))
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        with torch.no_grad():
            # batch size 1
            preds_list = []
            for i in range(input_signal.size(0)):
                preds, _, _ = sortformer_diar_model.forward(
                    input_signal[i : i + 1], input_signal_length[i : i + 1]
                )
                preds_list.append(preds)
            preds_instance = torch.cat(preds_list, 0)

            # batch size 4
            preds_batch, _, _ = sortformer_diar_model.forward(input_signal, input_signal_length)
        assert preds_instance.shape == preds_batch.shape

        diff = torch.mean(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
