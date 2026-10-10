"""Raw BEC sum diagnostics must respect graph boundaries and tensor bases."""

from unittest.mock import patch

import pytest
import torch
from e3nn.io import CartesianTensor

import sevenn._const as CONST
import sevenn._keys as KEY
from sevenn.error_recorder import (
    BECASRMaxError,
    BECASRMeanError,
    ErrorRecorder,
    MaximumNumber,
    get_err_type,
)


def output(cartesian, counts):
    # Predictions in the production model's nonsymmetric tensor irrep basis.
    return {
        KEY.PRED_BORN_EFFECTIVE_CHARGES: CartesianTensor('ij').from_cartesian(
            torch.tensor(cartesian, dtype=torch.float64)
        ),
        KEY.NUM_ATOMS: torch.tensor(counts),
    }


def test_asr_keeps_single_atom_and_unequal_graphs_separate():
    positive = [[1, 3, 0], [-2, 0, 0], [0, 0, 0]]
    negative = [[-0.5, -1.5, 0], [1, 0, 0], [0, 0, 0]]
    data = output([positive, negative, negative], [1, 2])
    maximum = BECASRMaxError(**get_err_type('BornEffectiveCharges'))
    mean = BECASRMeanError(**get_err_type('BornEffectiveCharges'))
    for metric in [maximum, mean]:
        metric.update(data)
        assert metric.get() == pytest.approx(3)
        # All three atomic tensors cancel when incorrectly summed over a batch.
        metric.update(output([[[0, 0, 0]] * 3], [1]))
    assert maximum.get() == pytest.approx(3)
    assert mean.get() == pytest.approx(2)  # Structure-weighted, not atom-weighted.
    maximum.reset()
    mean.reset()
    assert torch.isnan(torch.tensor(maximum.get()))
    assert torch.isnan(torch.tensor(mean.get()))
    for metric in [maximum, mean]:
        metric.update(output([[[0, 0, 0]] * 3], [1]))
        assert metric.get() == 0


def test_asr_config_and_recorder():
    records = [
        ['BornEffectiveCharges', 'ASRMean'],
        ['BornEffectiveCharges', 'ASRMax'],
    ]
    assert CONST.error_record_condition(records)
    assert not CONST.error_record_condition([['Force', 'ASRMax']])
    recorder = ErrorRecorder.from_config(
        {'error_record': records, 'is_train_stress': False}
    )
    recorder.update(output([[[1, 0, 0], [0, 2, 0], [0, 0, -4]]], [1]))
    values = recorder.get_metric_dict()
    assert values['BornEffectiveCharges_ASRMax (e)'] == pytest.approx(4)
    assert values['BornEffectiveCharges_ASRMean (e)'] == pytest.approx(4)
    recorder.epoch_forward()
    recorder.update(output([[[0, 0, 0]] * 3], [1]))
    assert recorder.get_metric_dict()['BornEffectiveCharges_ASRMax (e)'] == 0


def test_asr_max_ddp_uses_maximum_reduction():
    value = MaximumNumber()
    value.update(torch.tensor([2.0, 5.0]))

    def collective(tensor, op):
        if op == torch.distributed.ReduceOp.MAX:
            tensor.fill_(7)
        elif op == torch.distributed.ReduceOp.SUM:
            tensor.fill_(3)
        else:
            raise AssertionError(op)

    with patch('sevenn.error_recorder.dist.all_reduce', side_effect=collective):
        value._ddp_reduce('cpu')
    assert value.get() == 7
    assert value._count == 3


def test_monitor_does_not_change_predictions_or_random_state():
    data = output([[[1, 0, 0], [0, 2, 0], [0, 0, -4]]], [1])
    prediction = data[KEY.PRED_BORN_EFFECTIVE_CHARGES].clone()
    random_state = torch.random.get_rng_state().clone()
    metric = BECASRMaxError(**get_err_type('BornEffectiveCharges'))
    metric.update(data)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert torch.equal(prediction, data[KEY.PRED_BORN_EFFECTIVE_CHARGES])
