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
        '_target_': 'nemo.collections.asr.losses.bce_loss.BCELoss',
        'weight': None,
        'reduction': 'mean',
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
                preds = sortformer_diar_model.forward(input_signal[i : i + 1], input_signal_length[i : i + 1])
                preds_list.append(preds)
            preds_instance = torch.cat(preds_list, 0)

            # batch size 4
            preds_batch = sortformer_diar_model.forward(input_signal, input_signal_length)
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
            preds, activity_logits = model.forward_infer(hidden, hidden_lens, return_aux_logits=True)

        assert preds.shape == (2, 5, model.sortformer_modules.n_spk)
        assert activity_logits.shape == (2, 5, 3)

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
    def test_phantom_loss_penalizes_only_high_confidence_empty_channels(self, sortformer_model):
        num_spks = sortformer_model.sortformer_modules.n_spk
        targets_pil = torch.zeros(1, 4, num_spks)
        targets_pil[0, 0, 0] = 1.0
        target_lens = torch.tensor([3])

        one_phantom = torch.full_like(targets_pil, 0.1)
        # Activity on a non-empty channel is outside the scope of this loss.
        one_phantom[0, :3, 0] = 0.9
        # Two selected frames on one empty channel.
        one_phantom[0, 1, 1] = 0.5
        one_phantom[0, 2, 1] = 0.75
        # An empty channel below the threshold and activity in padding are ignored.
        one_phantom[0, 1, 2] = sortformer_model.phantom_threshold - 0.01
        one_phantom[0, 3, 3] = 0.9
        one_phantom.requires_grad_()

        one_phantom_loss = sortformer_model._phantom_loss(
            one_phantom, targets_pil, target_lens
        )
        expected_channel_loss = (-torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.25))) / 2
        assert torch.allclose(one_phantom_loss, expected_channel_loss / num_spks)

        # The fixed speaker-slot denominator makes a second identical phantom
        # channel add an equal amount to the loss.
        two_phantoms = one_phantom.detach().clone()
        two_phantoms[0, 1, 2] = 0.5
        two_phantoms[0, 2, 2] = 0.75
        two_phantom_loss = sortformer_model._phantom_loss(
            two_phantoms, targets_pil, target_lens
        )
        assert torch.allclose(two_phantom_loss, 2 * one_phantom_loss)

        one_phantom_loss.backward()
        assert torch.all(one_phantom.grad[0, 1:3, 1] > 0)
        assert torch.count_nonzero(one_phantom.grad[0, :, 0]) == 0
        assert one_phantom.grad[0, 1, 2] == 0
        assert torch.count_nonzero(one_phantom.grad[0, 3]) == 0

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
            'val_pairwise_ats_loss',
            'val_self_ats_loss',
            'val_spkcount_loss',
            'val_activity_loss',
            'val_presence_loss',
            'val_dice_loss',
            'val_phantom_loss',
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

        assert torch.equal(metrics['val_activity_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_presence_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_dice_loss'], torch.tensor(2.0))
        assert torch.equal(metrics['val_phantom_loss'], torch.tensor(2.0))


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
        monkeypatch.setattr(
            model,
            "forward_streaming",
            lambda processed_signal, processed_signal_length, return_aux_logits: torch.zeros_like(targets),
        )

        def fake_train_evaluations(preds, augmented, lengths, activity_logits=None):
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
            preds, activity_logits = model.forward(
                input_signal, input_signal_length, return_aux_logits=True
            )

        assert activity_logits.shape[:2] == preds.shape[:2]
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
                preds = sortformer_diar_model.forward(input_signal[i : i + 1], input_signal_length[i : i + 1])
                preds_list.append(preds)
            preds_instance = torch.cat(preds_list, 0)

            # batch size 4
            preds_batch = sortformer_diar_model.forward(input_signal, input_signal_length)
        assert preds_instance.shape == preds_batch.shape

        diff = torch.mean(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
