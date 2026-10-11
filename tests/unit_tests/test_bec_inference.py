"""BEC predictions survive unbatching and are exported in Cartesian order."""

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from torch_geometric.data import Batch

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.scripts.inference import write_inference_csv
from sevenn.util import to_atom_graph_list


def test_bec_csv_single_atom_and_unequal_batch(tmp_path):
    graphs = []
    for size in [1, 2]:
        graphs.append(
            AtomGraphData(
                **{
                    KEY.NUM_ATOMS: torch.tensor(size),
                    KEY.ATOMIC_NUMBERS: torch.ones(size, dtype=torch.long),
                    KEY.POS: torch.zeros(size, 3),
                    KEY.FORCE: torch.zeros(size, 3),
                    KEY.ENERGY: torch.tensor(0.0),
                    KEY.BORN_EFFECTIVE_CHARGES: torch.zeros(size, 3, 3),
                }
            )
        )
    batch = Batch.from_data_list(graphs)
    cartesian = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3) / 10
    ct = CartesianTensor('ij')
    batch[KEY.PRED_BORN_EFFECTIVE_CHARGES] = ct.from_cartesian(cartesian)
    batch[KEY.PRED_TOTAL_ENERGY] = torch.zeros(2)
    batch[KEY.ATOMIC_ENERGY] = torch.zeros(3)
    batch[KEY.PRED_FORCE] = torch.zeros(3, 3)
    separated = to_atom_graph_list(batch)
    assert separated[0][KEY.PRED_BORN_EFFECTIVE_CHARGES].shape == (1, 9)
    assert separated[1][KEY.PRED_BORN_EFFECTIVE_CHARGES].shape == (2, 9)
    write_inference_csv(separated, str(tmp_path))
    frame = pd.read_csv(tmp_path / 'per_atom.csv')
    columns = [
        f'inferred_born_effective_charges_{c}'
        for c in ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
    ]
    np.testing.assert_allclose(
        frame[columns].to_numpy().reshape(3, 3, 3), cartesian.numpy(), atol=1e-6
    )
    assert frame.stct_id.tolist() == [0, 1, 1]
    assert frame.atom_id.tolist() == [0, 0, 1]
