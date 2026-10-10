# Training example data

`data.xyz` contains 30 ZrO₂ structures, including oxygen-deficient cells, with DFT energies, forces, stress, and per-atom Born effective charges. The author confirmed on 2026-10-10 that this example is drawn from the official SevenNet-Polar datasets:

- Anh Khoa Augustin Lu, **Data set and SevenNet-Polar models for ZrO2, Li3PO4 and perovskites**, [Zenodo DOI: 10.5281/zenodo.21322761](https://doi.org/10.5281/zenodo.21322761). The archive identifies its data license as **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Attribute the author, dataset DOI, and accompanying paper when reusing the example.
- Lu et al., [SevenNet-Polar, arXiv:2607.14827](https://arxiv.org/abs/2607.14827), especially §4.7 for the dataset's VASP DFT/DFPT and PBE computational context. The tutorial does not reconstruct DFT settings from this extxyz file.

The source file first entered this repository in commit `14525a4`. It has no original frame IDs, trajectory IDs, or parent-structure labels. Tutorial partitions therefore demonstrate fitting/evaluation on a small sample; they do not establish trajectory-independent generalization. Research benchmarks should use known groups and independently held-out data from the archive.

Units: energy eV, force eV/Å, ASE stress eV/Å³, BEC e. BECs are flattened in row-major Cartesian order `xx xy xz yx yy yz zx zy zz`; the first index is polarization/field direction, the second displacement direction. Tutorial copies preserve the raw E/F/S/BEC labels and record a source-file SHA256 plus split indices. No sum-rule correction is applied.

Dielectric prediction is outside the scope of these tutorials. The unchanged source file includes an extra dielectric property on one frame; tutorial copies discard this optional field explicitly, and dielectric training is disabled.
