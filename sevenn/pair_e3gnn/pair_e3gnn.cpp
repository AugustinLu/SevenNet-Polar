/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Yutack Park (SNU)
------------------------------------------------------------------------- */

#include <ATen/ops/from_blob.h>
#include <array>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include "pair_e3gnn.h"

using namespace LAMMPS_NS;

// Undefined reference; body in pair_e3gnn_oeq_autograd.cpp to be linked
extern void pair_e3gnn_oeq_register_autograd();

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

PairE3GNN::PairE3GNN(LAMMPS *lmp) : Pair(lmp) {
  // constructor
  std::string device_name;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "PairE3GNN using device : %s\n", device_name.c_str());
  }
}

PairE3GNN::~PairE3GNN() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
    memory->destroy(elements);
  }
}

void PairE3GNN::compute(int eflag, int vflag) {
  // compute
  /*
     This compute function is ispired/modified from stress branch of pair-nequip
     https://github.com/mir-group/pair_nequip
  */

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;
  if (vflag_atom) {
    error->all(FLERR, "atomic stress is not supported\n");
  }

  int nlocal = list->inum; // same as nlocal
  int *ilist = list->ilist;
  tagint *tag = atom->tag;
  std::unordered_map<int, int> tag_map;

  if (atom->tag_consecutive() == 0) {
    for (int ii = 0; ii < nlocal; ii++) {
      const int i = ilist[ii];
      int itag = tag[i];
      tag_map[itag] = ii+1;
      // printf("MODIFY setting %i => %i \n",itag, tag_map[itag] );
    }
  } else {
    //Ordered which mappling required
    for (int ii = 0; ii < nlocal; ii++) {
        const int itag = ilist[ii]+1;
        tag_map[itag] = ii+1;
        // printf("normal setting %i => %i \n",itag, tag_map[itag] );
    }
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  long num_atoms[1] = {nlocal};

  int tag2i[nlocal];

  int *numneigh = list->numneigh;      // j loop cond
  int **firstneigh = list->firstneigh; // j list

  int bound;
  if (this->nedges_bound == -1) {
    bound = std::accumulate(numneigh, numneigh + nlocal, 0);
  } else {
    bound = this->nedges_bound;
  }
  const int nedges_upper_bound = bound;

  float cell[3][3];
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];
  cell[1][2] = 0.0;

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  torch::Tensor inp_cell = torch::from_blob(cell, {3, 3}, FLOAT_TYPE);
  torch::Tensor inp_num_atoms = torch::from_blob(num_atoms, {1}, INTEGER_TYPE);

  torch::Tensor inp_node_type = torch::zeros({nlocal}, INTEGER_TYPE);
  torch::Tensor inp_pos = torch::zeros({nlocal, 3});

  torch::Tensor inp_cell_volume =
      torch::dot(inp_cell[0], torch::cross(inp_cell[1], inp_cell[2], 0));

  float pbc_shift_tmp[nedges_upper_bound][3];

  auto node_type = inp_node_type.accessor<long, 1>();
  auto pos = inp_pos.accessor<float, 2>();

  long edge_idx_src[nedges_upper_bound];
  long edge_idx_dst[nedges_upper_bound];

  int nedges = 0;

  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist[ii];
    int itag = tag_map[tag[i]];
    tag2i[itag - 1] = i;
    const int itype = type[i];
    node_type[itag - 1] = map[itype];
    pos[itag - 1][0] = x[i][0];
    pos[itag - 1][1] = x[i][1];
    pos[itag - 1][2] = x[i][2];
  }

  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist[ii];
    int itag = tag_map[tag[i]];
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj]; // atom over pbc is different atom
      int jtag = tag_map[tag[j]]; // atom over pbs is same atom (it starts from 1)
      j &= NEIGHMASK;
      const int jtype = type[j];

      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1],
                               x[j][2] - x[i][2]};
      const double Rij =
          delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
      if (Rij < cutoff_square) {
        edge_idx_src[nedges] = itag - 1;
        edge_idx_dst[nedges] = jtag - 1;

        pbc_shift_tmp[nedges][0] = x[j][0] - pos[jtag - 1][0];
        pbc_shift_tmp[nedges][1] = x[j][1] - pos[jtag - 1][1];
        pbc_shift_tmp[nedges][2] = x[j][2] - pos[jtag - 1][2];

        nedges++;
      }
    } // j loop end
  }   // i loop end

  auto edge_idx_src_tensor =
      torch::from_blob(edge_idx_src, {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor =
      torch::from_blob(edge_idx_dst, {nedges}, INTEGER_TYPE);
  auto inp_edge_index =
      torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  // r' = r + {shift_tensor(integer vector of len 3)} @ cell_tensor
  // shift_tensor = (cell_tensor)^-1^T @ (r' - r)
  torch::Tensor cell_inv_tensor =
      inp_cell.inverse().transpose(0, 1).unsqueeze(0).to(device);
  torch::Tensor pbc_shift_tmp_tensor =
      torch::from_blob(pbc_shift_tmp, {nedges, 3}, FLOAT_TYPE)
          .view({nedges, 3, 1})
          .to(device);
  torch::Tensor inp_cell_shift =
      torch::bmm(cell_inv_tensor.expand({nedges, 3, 3}), pbc_shift_tmp_tensor)
          .view({nedges, 3});

  inp_pos.set_requires_grad(true);

  c10::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("pos", inp_pos.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));
  input_dict.insert("cell_lattice_vectors", inp_cell.to(device));
  input_dict.insert("cell_volume", inp_cell_volume.to(device));
  input_dict.insert("pbc_shift", inp_cell_shift);

  std::vector<torch::IValue> input(1, input_dict);
  auto output = model.forward(input).toGenericDict();

  torch::Tensor total_energy_tensor =
      output.at("inferred_total_energy").toTensor().cpu();
  torch::Tensor force_tensor = output.at("inferred_force").toTensor().cpu();
  auto forces = force_tensor.accessor<float, 2>();
  eng_vdwl += total_energy_tensor.item<float>();

  for (int itag = 0; itag < nlocal; itag++) {
    int i = tag2i[itag];
    f[i][0] += forces[itag][0];
    f[i][1] += forces[itag][1];
    f[i][2] += forces[itag][2];
  }

  if (has_efield && output.contains("inferred_born_effective_charges")) {
    torch::Tensor bec_tensor =
        output.at("inferred_born_effective_charges").toTensor().cpu();
    auto bec = bec_tensor.accessor<float, 2>();

    // Define mathematically exact constants at compile-time (64-bit precision)
    constexpr double inv_sqrt3 = 0.57735026918962576; // 1.0 / sqrt(3.0)
    constexpr double inv_sqrt2 = 0.70710678118654752; // 1.0 / sqrt(2.0)
    constexpr double inv_sqrt6 = 0.40824829046386302; // 1.0 / sqrt(6.0)
    constexpr double sqrt2_3   = 0.81649658092772603; // sqrt(2.0 / 3.0)

    // Reconstruct every atom's Cartesian BEC tensor first and, if
    // enforce_asr, subtract the mean so that Sum_i Z_i* = 0 exactly: the
    // field forces then sum to zero (no centre-of-mass drift). Mirrors
    // FieldCalculator.enforce_asr on the ASE side.
    std::vector<std::array<double, 9>> bec_cart(nlocal);
    std::array<double, 9> bec_mean = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int itag = 0; itag < nlocal; itag++) {
      double i0 = bec[itag][0];
      double i1 = bec[itag][1], i2 = bec[itag][2], i3 = bec[itag][3];
      double i4 = bec[itag][4], i5 = bec[itag][5], i6 = bec[itag][6];
      double i7 = bec[itag][7], i8 = bec[itag][8];

      // Reconstruct Cartesian tensor directly in double precision
      // Order: xx, xy, xz, yx, yy, yz, zx, zy, zz
      bec_cart[itag] = {
          inv_sqrt3 * i0 - inv_sqrt6 * i6 - inv_sqrt2 * i8,
          inv_sqrt2 * i3 + inv_sqrt2 * i5,
          -inv_sqrt2 * i2 + inv_sqrt2 * i4,
          -inv_sqrt2 * i3 + inv_sqrt2 * i5,
          inv_sqrt3 * i0 + sqrt2_3 * i6,
          inv_sqrt2 * i1 + inv_sqrt2 * i7,
          inv_sqrt2 * i2 + inv_sqrt2 * i4,
          -inv_sqrt2 * i1 + inv_sqrt2 * i7,
          inv_sqrt3 * i0 - inv_sqrt6 * i6 + inv_sqrt2 * i8,
      };
      if (enforce_asr) {
        for (int k = 0; k < 9; k++) bec_mean[k] += bec_cart[itag][k];
      }
    }
    if (enforce_asr && nlocal > 0) {
      for (int k = 0; k < 9; k++) bec_mean[k] /= nlocal;
    }

    for (int itag = 0; itag < nlocal; itag++) {
      int i = tag2i[itag];

      double c_xx = bec_cart[itag][0] - bec_mean[0];
      double c_xy = bec_cart[itag][1] - bec_mean[1];
      double c_xz = bec_cart[itag][2] - bec_mean[2];
      double c_yx = bec_cart[itag][3] - bec_mean[3];
      double c_yy = bec_cart[itag][4] - bec_mean[4];
      double c_yz = bec_cart[itag][5] - bec_mean[5];
      double c_zx = bec_cart[itag][6] - bec_mean[6];
      double c_zy = bec_cart[itag][7] - bec_mean[7];
      double c_zz = bec_cart[itag][8] - bec_mean[8];

      // Z*_ab = dP_a/dr_b: the FIRST index is the field index (the model's
      // convention -- see polar_output.py -- and the DFT labels'). The field
      // force is therefore F_b = sum_a Z*_ab E_a, i.e. F = Z*^T E: a COLUMN
      // of Z*, not a row. Earlier versions used the row (F = Z* E), which is
      // wrong for any non-symmetric Z* (about 10% of the field force, in the
      // transverse components, for ZrO2 at 0.05 V/A). The ASE
      // FieldCalculator uses the same convention (einsum 'iab,a->ib').
      double fx = (c_xx * efield[0] + c_yx * efield[1] + c_zx * efield[2]) * force->qe2f;
      double fy = (c_xy * efield[0] + c_yy * efield[1] + c_zy * efield[2]) * force->qe2f;
      double fz = (c_xz * efield[0] + c_yz * efield[1] + c_zz * efield[2]) * force->qe2f;

      if (update->ntimestep == update->firststep && tag[i] <= 3 && lmp->logfile) {
        fprintf(lmp->logfile, "Latest MD Step: %ld\n", static_cast<long>(update->ntimestep));
        fprintf(lmp->logfile, "Atom %d BEC Tensor (e):\n", static_cast<int>(tag[i]));
        fprintf(lmp->logfile, "[[ %11.8f  %11.8f  %11.8f]\n", c_xx, c_xy, c_xz);
        fprintf(lmp->logfile, " [ %11.8f  %11.8f  %11.8f]\n", c_yx, c_yy, c_yz);
        fprintf(lmp->logfile, " [ %11.8f  %11.8f  %11.8f]]\n", c_zx, c_zy, c_zz);
        fprintf(lmp->logfile, "Applied E-Field (V/A): [%g  %g  %g]\n", efield[0], efield[1], efield[2]);
        fprintf(lmp->logfile, "Resulting F_elec (eV/A): [%11.8f %11.8f %11.8f]\n\n", fx, fy, fz);
      }

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;

      // No field virial. The field force Z*^T E is not the gradient of any
      // energy (the Z*(R) field is not a Jacobian of a polarization function),
      // so there is no field energy whose strain derivative could define
      // one. fix npt under a field therefore sees only the model's zero-field
      // stress, as with LAMMPS's own fix efield by default.
    }
  }

  if (vflag) {
    // more accurately, it is virial part of stress
    torch::Tensor stress_tensor = output.at("inferred_stress").toTensor().cpu();
    auto virial_stress_tensor = stress_tensor * inp_cell_volume;
    // xy yz zx order in vasp (voigt is xx yy zz yz xz xy)
    auto virial_stress = virial_stress_tensor.accessor<float, 1>();
    virial[0] += virial_stress[0];
    virial[1] += virial_stress[1];
    virial[2] += virial_stress[2];
    virial[3] += virial_stress[3];
    virial[4] += virial_stress[5];
    virial[5] += virial_stress[4];
  }

  if (eflag_atom) {
    torch::Tensor atomic_energy_tensor =
        output.at("atomic_energy").toTensor().cpu().view({nlocal});
    auto atomic_energy = atomic_energy_tensor.accessor<float, 1>();
    for (int itag = 0; itag < nlocal; itag++) {
      int i = tag2i[itag];
      eatom[i] += atomic_energy[itag];
    }
  }

  // if it was the first MD step
  if (this->nedges_bound == -1) {
    this->nedges_bound = nedges * 1.2;
  } // else if the nedges is too small, increase the bound
  else if (nedges > this->nedges_bound / 1.2) {
    this->nedges_bound = nedges * 1.2;
  }
}

// allocate arrays (called from coeff)
void PairE3GNN::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(map, n + 1, "pair:map");
}

// global settings for pair_style
void PairE3GNN::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command");
  }
}

void PairE3GNN::coeff(int narg, char **arg) {

  if (allocated) {
    error->all(FLERR, "pair_e3gnn coeff called twice");
  }
  allocate();

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR,
               "e3gnn: first and second input of pair_coeff should be '*'");
  }
  // expected input : pair_coeff * * pot.pth type_name1 type_name2 ...

  std::unordered_map<std::string, std::string> meta_dict = {
      {"chemical_symbols_to_index", ""},
      {"cutoff", ""},
      {"num_species", ""},
      {"model_type", ""},
      {"version", ""},
      {"dtype", ""},
      {"flashTP", "version mismatch"},
      {"oeq", "version mismatch"},
      {"time", ""}};

  // model loading from input
  try {
    model = torch::jit::load(std::string(arg[2]), device, meta_dict);
  } catch (const c10::Error &e) {
    char err_msg[256];
    snprintf(err_msg, sizeof(err_msg), "error loading the model '%s': %s", arg[2], e.what_without_backtrace());
    error->all(FLERR, err_msg);
  }
  // model = torch::jit::freeze(model); model is already freezed

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy strategy;
  // thing about dynamic recompile as tensor shape varies, this is default
  // strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  strategy = {{torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);
  cutoff_square = cutoff * cutoff;

  // to make torch::autograd::grad() works
  if (meta_dict["oeq"] == "yes") {
    pair_e3gnn_oeq_register_autograd();
  }

  if (meta_dict["model_type"].compare("E3_equivariant_model") != 0) {
    error->all(FLERR, "given model type is not E3_equivariant_model");
  }

  std::string chem_str = meta_dict["chemical_symbols_to_index"];
  int ntypes = atom->ntypes;

  auto delim = " ";
  char *tok = std::strtok(const_cast<char *>(chem_str.c_str()), delim);
  std::vector<std::string> chem_vec;
  while (tok != nullptr) {
    chem_vec.push_back(std::string(tok));
    tok = std::strtok(nullptr, delim);
  }

  int chem_arg_i = 3;
  if (narg > chem_arg_i && strcmp(arg[chem_arg_i], "efield") == 0) {
    if (narg < chem_arg_i + 4) {
      error->all(FLERR, "e3gnn: efield keyword requires 3 values (ex ey ez)");
    }
    has_efield = true;
    efield[0] = std::stod(arg[chem_arg_i + 1]);
    efield[1] = std::stod(arg[chem_arg_i + 2]);
    efield[2] = std::stod(arg[chem_arg_i + 3]);
    chem_arg_i += 4;

    // Optional keywords after the field vector, in any order:
    //   enforce_asr [yes|no]   default yes (a bare "enforce_asr" means yes)
    while (narg > chem_arg_i) {
      if (strcmp(arg[chem_arg_i], "enforce_asr") == 0) {
        chem_arg_i += 1;
        if (narg > chem_arg_i && strcmp(arg[chem_arg_i], "no") == 0) {
          enforce_asr = false;
          chem_arg_i += 1;
        } else {
          enforce_asr = true;
          if (narg > chem_arg_i && strcmp(arg[chem_arg_i], "yes") == 0) chem_arg_i += 1;
        }
      } else if (strcmp(arg[chem_arg_i], "efield_virial") == 0) {
        error->all(FLERR, "e3gnn: efield_virial is not supported: the field force "
                          "Z*^T E derives from no energy, so there is no field virial");
      } else {
        break;
      }
    }
    if (lmp->logfile) {
      fprintf(lmp->logfile,
              "e3gnn: efield = (%g, %g, %g) V/A, enforce_asr %s, no field virial\n",
              efield[0], efield[1], efield[2], enforce_asr ? "yes" : "no");
    }
  }

  bool found_flag = false;
  int n_chem = narg - chem_arg_i;
  for (int i = 0; i < n_chem; i++) {
    found_flag = false;
    for (int j = 0; j < chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i + chem_arg_i]) == 0) {
        map[i + 1] = j;
        found_flag = true;
        if (lmp->logfile) {
          fprintf(lmp->logfile, "Chemical specie '%s' is assigned to type %d\n",
                  arg[i + chem_arg_i], i + 1);
        }
        break;
      }
    }
    if (!found_flag) {
      error->all(FLERR, "Unknown chemical specie is given");
    }
  }

  if (ntypes > n_chem) {
    error->all(FLERR, "Not enough chemical specie is given. Check pair_coeff "
                      "and types in your data/script");
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = 1; j <= ntypes; j++) {
      if ((map[i] >= 0) && (map[j] >= 0)) {
        setflag[i][j] = 1;
        cutsq[i][j] = cutoff * cutoff;
      }
    }
  }

  if (lmp->logfile) {
    fprintf(lmp->logfile, "from sevenn version '%s' ",
            meta_dict["version"].c_str());
    fprintf(lmp->logfile, "%s precision model, deployed: %s\n",
            meta_dict["dtype"].c_str(), meta_dict["time"].c_str());
    fprintf(lmp->logfile, "FlashTP: %s\n",
            meta_dict["flashTP"].c_str());
    fprintf(lmp->logfile, "OEQ: %s\n",
            meta_dict["oeq"].c_str());
  }
}

// init specific to this pair
void PairE3GNN::init_style() {
  // Newton flag is irrelevant if use only one processor for simulation
  /*
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style nn requires newton pair on");
  }
  */

  // full neighbor list (this is many-body potential)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNN::init_one(int i, int j) { return cutoff; }
