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


class TestSortformerEncLabelModelStreaming:
    @pytest.mark.unit
    def test_constructor(self, sortformer_model):
        sortformer_model.streaming_mode = True
        sortformer_diar_model = sortformer_model.train()
        confdict = sortformer_diar_model.to_config_dict()
        instance2 = SortformerEncLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, SortformerEncLabelModel)

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
