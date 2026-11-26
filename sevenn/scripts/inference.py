import csv
import os
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.train.modal_dataset import SevenNetMultiModalDataset


def _bec_to_cartesian(bec):
    """
    Predicted BEC in irreps form (N, 9), 1x0e+1x1e+1x2e, to Cartesian (N, 3, 3),
    Z*_ab with the field index first, as written by the ASE calculator.
    Input that is not in this form is returned unchanged.
    """
    bec = np.asarray(bec)
    if bec.ndim == 1 and bec.size == 9:  # one atom, squeezed by fit_dimension
        bec = bec.reshape(1, 9)
    if bec.ndim != 2 or bec.shape[-1] != 9:
        return bec
    from e3nn.io import CartesianTensor

    ct = CartesianTensor('ij')
    rtp = ct.reduced_tensor_products()
    bec_t = torch.as_tensor(bec)
    return ct.to_cartesian(bec_t, rtp.to(bec_t.dtype)).numpy()


def write_inference_csv(output_list, out):
    for i, output in enumerate(output_list):
        output = output.fit_dimension()
        if KEY.STRESS in output:
            output[KEY.STRESS] = output[KEY.STRESS] * 1602.1766208
        if KEY.PRED_STRESS in output:
            output[KEY.PRED_STRESS] = output[KEY.PRED_STRESS] * 1602.1766208
        output_list[i] = output.to_numpy_dict()

    per_graph_keys = [
        KEY.NUM_ATOMS,
        KEY.USER_LABEL,
        KEY.ENERGY,
        KEY.PRED_TOTAL_ENERGY,
        KEY.STRESS,
        KEY.PRED_STRESS,
        KEY.DIELECTRIC_TENSOR,
        KEY.PRED_DIELECTRIC_TENSOR,
    ]

    per_atom_keys = [
        KEY.ATOMIC_NUMBERS,
        KEY.ATOMIC_ENERGY,
        KEY.POS,
        KEY.FORCE,
        KEY.PRED_FORCE,
        KEY.BORN_EFFECTIVE_CHARGES,
        KEY.PRED_BORN_EFFECTIVE_CHARGES,
    ]

    def unfold_dct_val(dct, keys, suffix_list=None, sfx_dict=None):
        res = {}
        if suffix_list is None:
            suffix_list = range(100)
        if sfx_dict is None:
            sfx_dict = {}

        for k in keys:
            if k not in dct:
                res[k] = '-'
            elif isinstance(dct[k], np.ndarray) and dct[k].ndim != 0:
                sfx = sfx_dict.get(k, suffix_list)
                res.update(
                    {f'{k}_{sfx[i]}': v for i, v in enumerate(dct[k].flatten())}
                )
            else:
                res[k] = dct[k]
        return res

    def per_atom_dct_list(dct, keys):
        sfx_list = ['x', 'y', 'z']
        sfx_dict = {
            KEY.BORN_EFFECTIVE_CHARGES: [
                'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'
            ],
            KEY.PRED_BORN_EFFECTIVE_CHARGES: [
                'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'
            ],
        }
        res = []
        natoms = dct[KEY.NUM_ATOMS]
        extracted = {k: dct[k] for k in keys if k in dct}
        if KEY.PRED_BORN_EFFECTIVE_CHARGES in extracted:
            extracted[KEY.PRED_BORN_EFFECTIVE_CHARGES] = _bec_to_cartesian(
                extracted[KEY.PRED_BORN_EFFECTIVE_CHARGES]
            )
        for i in range(natoms):
            raw = {}
            raw.update({k: v[i] for k, v in extracted.items()})
            per_atom_dct = unfold_dct_val(
                raw, keys, suffix_list=sfx_list, sfx_dict=sfx_dict
            )
            res.append(per_atom_dct)
        return res

    try:
        with open(f'{out}/info.csv', 'w', newline='') as f:
            header = output_list[0][KEY.INFO].keys()
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for output in output_list:
                writer.writerow(output[KEY.INFO])
    except (KeyError, TypeError, AttributeError, csv.Error) as e:
        print(e)
        print('failed to write meta data, info.csv is not written')

    with open(f'{out}/per_graph.csv', 'w', newline='') as f:
        sfx_list = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']  # default for stress
        sfx_dict = {
            KEY.DIELECTRIC_TENSOR: [
                'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'
            ],
            KEY.PRED_DIELECTRIC_TENSOR: [
                'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'
            ],
        }
        writer = None
        for output in output_list:
            data = unfold_dct_val(
                output, per_graph_keys, suffix_list=sfx_list, sfx_dict=sfx_dict
            )

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writeheader()
            writer.writerow(data)

    with open(f'{out}/per_atom.csv', 'w', newline='') as f:
        writer = None
        for i, output in enumerate(output_list):
            list_of_dct = per_atom_dct_list(output, per_atom_keys)
            for j, dct in enumerate(list_of_dct):
                idx_dct = {'stct_id': i, 'atom_id': j}
                data = {**idx_dct, **dct}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    writer.writeheader()
                writer.writerow(data)


def project_bec_asr(output: AtomGraphData) -> None:
    """
    Enforce the acoustic sum rule on the predicted BEC of a (batched) output:
    subtract the mean predicted BEC of each structure, so that
    sum_i Z*_i = 0 for every structure. Same projection as
    FieldCalculator(enforce_asr=True) and the LAMMPS ``enforce_asr`` keyword.
    The BEC is in its irreps form here; the map to Cartesian tensors is
    linear, so subtracting the mean in either form gives the same result.
    """
    key = KEY.PRED_BORN_EFFECTIVE_CHARGES
    if key not in output:
        raise ValueError('enforce_asr requires a model that predicts BEC')
    bec = output[key]
    flat = bec.reshape(bec.shape[0], -1)
    if KEY.BATCH in output:
        batch = output[KEY.BATCH]
    else:
        batch = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
    n_graph = int(batch.max().item()) + 1 if flat.shape[0] > 0 else 0
    sums = torch.zeros(
        (n_graph, flat.shape[1]), dtype=flat.dtype, device=flat.device
    ).index_add_(0, batch, flat)
    counts = torch.bincount(batch, minlength=n_graph).to(flat.dtype).unsqueeze(-1)
    output[key] = (flat - (sums / counts)[batch]).reshape(bec.shape)


def _error_recorder(output: AtomGraphData):
    recorder_tuples = [
        ('Energy', 'RMSE'),
        ('Force', 'RMSE'),
        ('Stress', 'RMSE'),
        ('Energy', 'MAE'),
        ('Force', 'MAE'),
        ('Stress', 'MAE'),
    ]
    if KEY.PRED_BORN_EFFECTIVE_CHARGES in output:
        # unlabeled BEC are NaN and are skipped by the metrics
        recorder_tuples += [
            ('BornEffectiveCharges', 'DiagRMSE'),
            ('BornEffectiveCharges', 'OffDiagRMSE'),
        ]
    return util.get_error_recorder(recorder_tuples)


def _patch_data_info(
    graph_list: Iterable[AtomGraphData], full_file_list: List[str]
) -> None:
    keys = set()
    for graph, path in zip(graph_list, full_file_list):
        if KEY.INFO not in graph:
            graph[KEY.INFO] = {}
        graph[KEY.INFO].update({'file': os.path.abspath(path)})
        keys.update(graph[KEY.INFO].keys())

    # save only safe subset of info (for batching)
    for graph in graph_list:
        info_dict = graph[KEY.INFO]
        info_dict.update({k: '' for k in keys if k not in info_dict})


def inference(
    checkpoint: str,
    targets: Union[str, List[str]],
    output_dir: str,
    num_workers: int = 1,
    device: str = 'cpu',
    batch_size: int = 4,
    save_graph: bool = False,
    allow_unlabeled: bool = False,
    modal: Optional[str] = None,
    enable_cueq: bool = False,
    enable_flash: bool = False,
    enable_oeq: bool = False,
    enforce_asr: bool = False,
    **data_kwargs,
) -> None:
    """
    Inference model on the target dataset, writes
    per_graph, per_atom inference results in csv format
    to the output_dir
    If a given target doesn't have EFS key, it puts dummy
    values.

    Args:
        checkpoint: model checkpoint path,
        target: path, or list of path to evaluate. Supports
            ASE readable, sevenn_data/*.pt, .sevenn_data, and
            structure_list
        output_dir: directory to write results
        num_workers: number of workers to build graph
        device: device to evaluate, defaults to 'auto'
        batch_size: batch size for inference
        save_grpah: if True, save preprocessed graph to output dir
        enforce_asr: if True, subtract the mean predicted BEC of each
            structure (acoustic sum rule, sum_i Z*_i = 0) before errors and
            outputs are computed. Off by default. Do not use it for charged
            cells, where sum_i Z*_i = Q I.
        data_kwargs: keyword arguments used when reading targets,
            for example, given index='-1', only the last snapshot
            will be evaluated if it was ASE readable.
            While this function can handle different types of targets
            at once, it will not work smoothly with data_kwargs

    """
    # TODO: False as default, priority?
    model, _ = util.model_from_checkpoint(
        checkpoint,
        enable_cueq=enable_cueq or None,
        enable_flash=enable_flash or None,
        enable_oeq=enable_oeq or None,
    )
    cutoff = model.cutoff

    if modal:
        if model.modal_map is None:
            raise ValueError('Modality given, but model has no modal_map')
        if modal not in model.modal_map:
            _modals = list(model.modal_map.keys())
            raise ValueError(f'Unknown modal {modal} (not in {_modals})')

    if isinstance(targets, str):
        targets = [targets]

    full_file_list = []
    if save_graph:
        dataset = SevenNetGraphDataset(
            cutoff=cutoff,
            root=output_dir,
            files=targets,
            process_num_cores=num_workers,
            processed_name='saved_graph.pt',
            **data_kwargs,
        )
        full_file_list = dataset.full_file_list  # TODO: not used currently
    else:
        dataset = []
        for file in targets:
            tmplist = SevenNetGraphDataset.file_to_graph_list(
                file,
                cutoff=cutoff,
                num_cores=num_workers,
                allow_unlabeled=allow_unlabeled,
                **data_kwargs,
            )
            dataset.extend(tmplist)
            full_file_list.extend([os.path.abspath(file)] * len(tmplist))
    if (
        full_file_list is not None
        and len(full_file_list) == len(dataset)
        and not isinstance(dataset, SevenNetGraphDataset)
    ):
        _patch_data_info(dataset, full_file_list)  # type: ignore

    if modal:
        dataset = SevenNetMultiModalDataset({modal: dataset})  # type: ignore

    loader = DataLoader(dataset, batch_size, shuffle=False)  # type: ignore

    model.to(device)
    model.set_is_batch_data(True)
    model.eval()

    rec = None
    output_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        output = model(batch).detach().cpu()
        if enforce_asr:
            project_bec_asr(output)
        if rec is None:
            rec = _error_recorder(output)
        rec.update(output)
        output_list.extend(util.to_atom_graph_list(output))

    errors = rec.epoch_forward() if rec is not None else {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'errors.txt'), 'w', encoding='utf-8') as f:
        for key, val in errors.items():
            f.write(f'{key}: {val}\n')

    write_inference_csv(output_list, output_dir)
