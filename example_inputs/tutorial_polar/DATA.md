# Official data subsets for tutorial experiments

These bounded pools contain **240 structures per material**, sampled from the
author's [SevenNet-Polar dataset release](https://doi.org/10.5281/zenodo.21322761):
10,103 ZrO₂ and 17,900 Li₃PO₄ configurations. The original complete files are
not included here. The source checksums match the official Zenodo files.

Each pool has an ordered training reservoir of 200 structures, plus **20 fixed
validation and 20 fixed test structures**. Taking a prefix of the reservoir gives
nested training subsets. Selection uses seed `20261010` and balanced composition
coverage, without inspecting BEC values, ASR residuals, or model errors. Model
optimization uses its own seed. Whole structures are kept intact.

The manifests record the original source-file SHA256, source frame IDs, pool
checksum, selected compositions, and geometry fingerprints. Exact ordered
geometry screening found no duplicates in either full source file; the selected
pools additionally pass a permutation/translation/rotation invariant geometry
screen. Trajectory/parent IDs are unavailable, so these demonstration splits do
not establish independent scientific generalization.

Extended XYZ serialization preserves energies, forces, stress and all nine BEC
components, with numerical round-trip checks. No acoustic-sum-rule projection,
label centering, or filtering by label quality is applied. Optional dielectric
labels in the ZrO₂ source are excluded from these tutorial copies.

## Reference tensor sums

All ZrO₂ compositions have reference maximum tensor-sum components below about
`1.2e-7 e` in the original files. Li₁₂O₁₆P₄ and Li₂₂O₃₁P₈ reference sums are
small but nonzero. Li₁₁O₁₆P₄ sums are approximately `1.02 e` in magnitude,
consistent with the dataset author's explanation that the Li-vacancy DFPT was
probably performed with a charged cell. That convention has not been
independently established from calculation inputs. The labels are retained as
supplied, including their nonzero sums.

For Li₃PO₄, inspect both the raw predicted sum and the prediction-minus-reference
tensor sum for each structure. A raw maximum near `1 e` is not by itself evidence
of poor agreement with these labels. Subtracting partition-level scalar maxima
does not yield the tensor-sum prediction error.

## Attribution and license

These data are distributed under **CC BY 4.0**, as specified by the Zenodo record.
Please cite the record, the SevenNet-Polar paper, and the source-material paper:

- Lu et al. (2026), *SevenNet-Polar for MultiTask Prediction of Energy, Forces,
  Stress, and Born Effective Charges*, [arXiv:2607.14827](https://arxiv.org/abs/2607.14827).
- ZrO₂: Lu et al. (2026), *Study of the ion mobility in defect-laden ZrO₂ under
  an electric field using neural network with predictions for Born effective
  charges*, [Physical Review Materials 10, 066001](https://doi.org/10.1103/jcsd-dbl2).
- Li₃PO₄: Shimizu et al. (2023), *Prediction of Born effective charges using
  neural network to study ion migration under electric fields: Applications
  to crystalline and amorphous Li₃PO₄*,
  [STAM Methods 3, 2253135](https://doi.org/10.1080/27660400.2023.2253135).

The smaller 30-structure example and its attribution remain in
[`../training_polar`](../training_polar/DATA.md).
