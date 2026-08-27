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

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/ops/from_blob.h>
#include <array>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

#include <mpi.h>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "comm_brick.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
// #include "nvToolsExt.h"

#include "pair_e3gnn_parallel.h"
#include <cassert>

#ifdef OMPI_MPI_H
#include "mpi-ext.h" //This should be included after mpi.h which is included in pair.h
#endif

using namespace LAMMPS_NS;

// Undefined reference; body in pair_e3gnn_oeq_autograd.cpp to be linked
extern void pair_e3gnn_oeq_register_autograd();

#define INTEGER_TYPE torch::TensorOptions().dtype(torch::kInt64)
#define FLOAT_TYPE torch::TensorOptions().dtype(torch::kFloat)

DeviceBuffManager &DeviceBuffManager::getInstance() {
  static DeviceBuffManager instance;
  return instance;
}

void DeviceBuffManager::get_buffer(int send_size, int recv_size,
                                   float *&buf_send_ptr, float *&buf_recv_ptr) {
  if (send_size > send_buf_size) {
    cudaFree(buf_send_device);
    cudaError_t cuda_err =
        cudaMalloc(&buf_send_device, send_size * sizeof(float));
    send_buf_size = send_size;
  }
  if (recv_size > recv_buf_size) {
    cudaFree(buf_recv_device);
    cudaError_t cuda_err =
        cudaMalloc(&buf_recv_device, recv_size * sizeof(float));
    recv_buf_size = recv_size;
  }
  buf_send_ptr = buf_send_device;
  buf_recv_ptr = buf_recv_device;
}

DeviceBuffManager::~DeviceBuffManager() {
  cudaFree(buf_send_device);
  cudaFree(buf_recv_device);
}

PairE3GNNParallel::PairE3GNNParallel(LAMMPS *lmp) : Pair(lmp) {
  // constructor
  world_rank = comm->me;

  std::string device_name;
  const bool use_gpu = torch::cuda::is_available();

  comm_forward = 0;
  comm_reverse = 0;

  // OpenMPI detection
#ifdef OMPI_MPI_H
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 == MPIX_Query_cuda_support()) {
    use_cuda_mpi = true;
  } else {
    use_cuda_mpi = false;
  }
#else
  use_cuda_mpi = false;
#endif
#else
  use_cuda_mpi = false;
#endif
  // use_cuda_mpi = use_gpu && use_cuda_mpi;
  // if (use_cuda_mpi) {
  if (use_gpu) {
    device = get_cuda_device();
    device_name = "CUDA";
  } else {
    device = torch::kCPU;
    device_name = "CPU";
  }

  if (std::getenv("OFF_E3GNN_PARALLEL_CUDA_MPI")) {
      use_cuda_mpi = false;
  }

  if (lmp->screen) {
    if (use_gpu && !use_cuda_mpi) {
      device_comm = torch::kCPU;
      fprintf(lmp->screen,
              "cuda-aware mpi not found, communicate via host device\n");
    } else {
      device_comm = device;
    }
    fprintf(lmp->screen, "PairE3GNNParallel using device : %s\n",
            device_name.c_str());
    fprintf(lmp->screen, "PairE3GNNParallel cuda-aware mpi: %s\n",
            use_cuda_mpi ? "True" : "False");
  }
  if (lmp->logfile) {
    if (use_gpu && !use_cuda_mpi) {
      device_comm = torch::kCPU;
      fprintf(lmp->logfile,
              "cuda-aware mpi not found, communicate via host device\n");
    } else {
      device_comm = device;
    }
    fprintf(lmp->logfile, "PairE3GNNParallel using device : %s\n",
            device_name.c_str());
    fprintf(lmp->logfile, "PairE3GNNParallel cuda-aware mpi: %s\n",
            use_cuda_mpi ? "True" : "False");
  }
}

torch::Device PairE3GNNParallel::get_cuda_device() {
  char *cuda_visible = std::getenv("CUDA_VISIBLE_DEVICES");
  int num_gpus;
  int idx;
  int rank = comm->me;
  num_gpus = torch::cuda::device_count();
  idx = rank % num_gpus;
  cudaError_t cuda_err = cudaSetDevice(idx);
  if (cuda_err != cudaSuccess) {
    std::cerr << "E3GNN: Failed to set CUDA device: "
              << cudaGetErrorString(cuda_err) << std::endl;
  }
  return torch::Device(torch::kCUDA, idx);
}

PairE3GNNParallel::~PairE3GNNParallel() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
  }
}

int PairE3GNNParallel::get_x_dim() { return x_dim; }

bool PairE3GNNParallel::use_cuda_mpi_() { return use_cuda_mpi; }

bool PairE3GNNParallel::is_comm_preprocess_done() {
  return comm_preprocess_done;
}

void PairE3GNNParallel::compute(int eflag, int vflag) {
  /*
     Graph build on cpu
  */
  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;
  if (vflag_atom) {
    error->all(FLERR, "atomic stress is not supported\n");
  }

  if (atom->tag_consecutive() == 0) {
    error->all(FLERR, "Pair e3gnn requires consecutive atom IDs");
  }

  double **x = atom->x;
  double **f = atom->f;
  imageint *image = atom->image;
  int *type = atom->type;
  int nlocal = list->inum; // same as nlocal
  int nghost = atom->nghost;
  int ntotal = nlocal + nghost;
  int *ilist = list->ilist;
  int inum = list->inum;

  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);
  if (comm_brick == nullptr) {
    error->all(FLERR, "e3gnn/parallel: comm style should be brick & from "
                      "modified code of comm_brick");
  }

  bigint natoms = atom->natoms;

  // tag ignore PBC
  tagint *tag = atom->tag;

  // store graph_idx from local to known ghost atoms(ghost atoms inside cutoff)
  int tag_to_graph_idx[natoms + 1]; // tag starts from 1 not 0
  std::fill_n(tag_to_graph_idx, natoms + 1, -1);

  // to access tag_to_graph_idx from comm
  tag_to_graph_idx_ptr = tag_to_graph_idx;

  int graph_indexer = nlocal;
  int graph_index_to_i[ntotal];

  int *numneigh = list->numneigh;      // j loop cond
  int **firstneigh = list->firstneigh; // j list
  const int nedges_upper_bound =
      std::accumulate(numneigh, numneigh + nlocal, 0);

  std::vector<long> node_type;
  std::vector<long> node_type_ghost;

  float edge_vec[nedges_upper_bound][3];
  long edge_idx_src[nedges_upper_bound];
  long edge_idx_dst[nedges_upper_bound];

  int nedges = 0;
  for (int ii = 0; ii < inum; ii++) {
    // populate tag_to_graph_idx of local atoms
    const int i = ilist[ii];
    const int itag = tag[i];
    const int itype = type[i];
    tag_to_graph_idx[itag] = ii;
    graph_index_to_i[ii] = i;
    node_type.push_back(map[itype]);
  }

  // loop over neighbors, build graph
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int i_graph_idx = ii;
    const int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const int jtag = tag[j];
      j &= NEIGHMASK;
      const int jtype = type[j];
      // we have to calculate Rij to check cutoff in lammps side
      const double delij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1],
                               x[j][2] - x[i][2]};
      const double Rij =
          delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];

      int j_graph_idx;
      if (Rij < cutoff_square) {
        // if given j is not local atom and inside cutoff
        if (tag_to_graph_idx[jtag] == -1) {
          // if j is ghost atom inside cutoff but first seen
          tag_to_graph_idx[jtag] = graph_indexer;
          graph_index_to_i[graph_indexer] = j;
          node_type_ghost.push_back(map[jtype]);
          graph_indexer++;
        }

        j_graph_idx = tag_to_graph_idx[jtag];
        edge_idx_src[nedges] = i_graph_idx;
        edge_idx_dst[nedges] = j_graph_idx;
        edge_vec[nedges][0] = delij[0];
        edge_vec[nedges][1] = delij[1];
        edge_vec[nedges][2] = delij[2];
        nedges++;
      }
    } // j loop end
  }   // i loop end

  // member variable
  graph_size = graph_indexer;
  const int ghost_node_num = graph_size - nlocal;

  // convert data to Tensor
  auto inp_node_type = torch::from_blob(node_type.data(), nlocal, INTEGER_TYPE);
  auto inp_node_type_ghost =
      torch::from_blob(node_type_ghost.data(), ghost_node_num, INTEGER_TYPE);

  long num_nodes[1] = {long(nlocal)};
  auto inp_num_atoms = torch::from_blob(num_nodes, {1}, INTEGER_TYPE);

  auto edge_idx_src_tensor =
      torch::from_blob(edge_idx_src, {nedges}, INTEGER_TYPE);
  auto edge_idx_dst_tensor =
      torch::from_blob(edge_idx_dst, {nedges}, INTEGER_TYPE);
  auto inp_edge_index =
      torch::stack({edge_idx_src_tensor, edge_idx_dst_tensor});

  auto inp_edge_vec = torch::from_blob(edge_vec, {nedges, 3}, FLOAT_TYPE);

  // r_original requires grad True
  inp_edge_vec.set_requires_grad(true);

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
  torch::Tensor inp_cell_volume =
      torch::dot(inp_cell[0], torch::cross(inp_cell[1], inp_cell[2], 0));

  torch::Dict<std::string, torch::Tensor> input_dict;
  input_dict.insert("x", inp_node_type.to(device));
  input_dict.insert("x_ghost", inp_node_type_ghost.to(device));
  input_dict.insert("edge_index", inp_edge_index.to(device));
  input_dict.insert("edge_vec", inp_edge_vec.to(device));
  input_dict.insert("pos", torch::zeros({ntotal, 3}).to(device));
  input_dict.insert("num_atoms", inp_num_atoms.to(device));
  input_dict.insert("nlocal", inp_num_atoms.to(torch::kCPU));
  input_dict.insert("cell_volume", inp_cell_volume.to(device));

  std::list<std::vector<torch::Tensor>> wrt_tensors;
  wrt_tensors.push_back({input_dict.at("edge_vec")});

  auto model_part = model_list.front();

  auto output = model_part.forward({input_dict}).toGenericDict();

  comm_preprocess();

  // extra_graph_idx_map is set from comm_preprocess();
  // last one is for trash values. See pack_forward_init
  const int extra_size =
      ghost_node_num + static_cast<int>(extra_graph_idx_map.size()) + 1;
  torch::Tensor x_local;
  torch::Tensor x_ghost;

  for (auto it = model_list.begin(); it != model_list.end(); ++it) {
    if (it == model_list.begin())
      continue;
    model_part = *it;

    x_local = output.at("x").toTensor().detach().to(device);
    x_dim = x_local.size(1); // length of per atom vector(node feature)

    auto ghost_and_extra_x = torch::zeros({ghost_node_num + extra_size, x_dim},
                                          FLOAT_TYPE.device(device));
    x_comm = torch::cat({x_local, ghost_and_extra_x}, 0).to(device_comm);
    comm_brick->forward_comm(this); // populate x_ghost by communication

    // What we got from forward_comm (node feature of ghosts)
    x_ghost = torch::split_with_sizes(
        x_comm, {nlocal, ghost_node_num, extra_size}, 0)[1];
    x_ghost.set_requires_grad(true);

    // prepare next input (output > next input)
    output.insert_or_assign("x_ghost", x_ghost.to(device));
    // make another edge_vec to discriminate grad calculation with other
    // edge_vecs(maybe redundant?)
    output.insert_or_assign("edge_vec",
                            output.at("edge_vec").toTensor().clone());

    // save tensors for backprop
    wrt_tensors.push_back({output.at("edge_vec").toTensor(),
                           output.at("x").toTensor(),
                           output.at("self_cont_tmp").toTensor(),
                           output.at("x_ghost").toTensor()});

    output = model_part.forward({output}).toGenericDict();
  }
  torch::Tensor energy_tensor =
      output.at("inferred_total_energy").toTensor().squeeze();

  torch::Tensor dE_dr =
      torch::zeros({nedges, 3}, FLOAT_TYPE.device(device)); // create on device
  torch::Tensor x_local_save; // holds grad info of x_local (it loses its grad
                              // when sends to CPU)
  torch::Tensor self_conn_grads;
  std::vector<torch::Tensor> grads;
  std::vector<torch::Tensor> of_tensor;

  // TODO: most values of self_conn_grads were zero because we use only scalars
  // for energy
  for (auto rit = wrt_tensors.rbegin(); rit != wrt_tensors.rend(); ++rit) {
    // edge_vec, x, x_ghost order
    auto wrt_tensor = *rit;
    if (rit == wrt_tensors.rbegin()) {
      grads = torch::autograd::grad({energy_tensor}, wrt_tensor);
    } else {
      x_local_save.copy_(x_local);
      //                            of         wrt         grads_output
      grads = torch::autograd::grad(of_tensor, wrt_tensor,
                                    {x_local_save, self_conn_grads});
    }

    dE_dr = dE_dr + grads.at(0); // accumulate force
    if (std::distance(rit, wrt_tensors.rend()) == 1)
      continue; // if last iteration

    of_tensor.clear();
    of_tensor.push_back(wrt_tensor[1]); // x
    of_tensor.push_back(wrt_tensor[2]); // self_cont_tmp

    x_local_save = grads.at(1);      // for grads_output
    x_local = x_local_save.detach(); // grad_outputs & communication
    x_dim = x_local.size(1);

    self_conn_grads = grads.at(2); // no communication, for grads_output

    x_ghost = grads.at(3).detach(); // yes communication, not for grads_output

    auto extra_x = torch::zeros({extra_size, x_dim}, FLOAT_TYPE.device(device));
    x_comm = torch::cat({x_local, x_ghost, extra_x}, 0).to(device_comm);

    comm_brick->reverse_comm(this); // completes x_local

    // now x_local is complete (dE_dx), become next grads_output(with
    // self_conn_grads)
    x_local = torch::split_with_sizes(
        x_comm, {nlocal, ghost_node_num, extra_size}, 0)[0];
  }

  eng_vdwl += energy_tensor.item<float>(); // accumulate energy

  dE_dr = dE_dr.to(torch::kCPU);
  torch::Tensor force_tensor = torch::zeros({graph_indexer, 3});

  auto _edge_idx_src_tensor =
      edge_idx_src_tensor.repeat_interleave(3).view({nedges, 3});
  auto _edge_idx_dst_tensor =
      edge_idx_dst_tensor.repeat_interleave(3).view({nedges, 3});

  force_tensor.scatter_reduce_(0, _edge_idx_src_tensor, dE_dr, "sum");
  force_tensor.scatter_reduce_(0, _edge_idx_dst_tensor, torch::neg(dE_dr),
                               "sum");

  auto forces = force_tensor.accessor<float, 2>();

  for (int graph_idx = 0; graph_idx < graph_indexer; graph_idx++) {
    int i = graph_index_to_i[graph_idx];
    f[i][0] += forces[graph_idx][0];
    f[i][1] += forces[graph_idx][1];
    f[i][2] += forces[graph_idx][2];
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

    // Reconstruct every atom's Cartesian BEC tensor first (needed either
    // way), and, if enforce_asr, subtract the mean so that Sum_i Z_i* = 0
    // exactly before it is used for both the force and the virial -- keeps
    // them mutually consistent derivatives of the same (ASR-clean)
    // quantity. Mirrors FieldCalculator.enforce_asr on the ASE side.
    std::vector<std::array<double, 9>> bec_cart(nlocal);
    std::array<double, 9> bec_mean = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int graph_idx = 0; graph_idx < nlocal; graph_idx++) {
      double i0 = bec[graph_idx][0];
      double i1 = bec[graph_idx][1], i2 = bec[graph_idx][2], i3 = bec[graph_idx][3];
      double i4 = bec[graph_idx][4], i5 = bec[graph_idx][5], i6 = bec[graph_idx][6];
      double i7 = bec[graph_idx][7], i8 = bec[graph_idx][8];

      // Order: xx, xy, xz, yx, yy, yz, zx, zy, zz
      bec_cart[graph_idx] = {
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
        for (int k = 0; k < 9; k++) bec_mean[k] += bec_cart[graph_idx][k];
      }
    }
    if (enforce_asr) {
      // bec_mean currently holds this rank's LOCAL sum over its own owned
      // atoms only. The acoustic sum rule is a property of the whole
      // physical system (Sum over ALL atoms of Z_i* = 0), not of whatever
      // subset domain decomposition happened to assign to this rank -- a
      // per-rank local mean would make the correction (and therefore the
      // force and virial) depend on the MPI rank count and processor
      // decomposition, which must not happen. Reduce to the true global
      // sum, then divide by the global atom count (already available,
      // computed once above from atom->natoms -- not another reduction).
      std::array<double, 9> bec_sum_global = {0, 0, 0, 0, 0, 0, 0, 0, 0};
      MPI_Allreduce(bec_mean.data(), bec_sum_global.data(), 9, MPI_DOUBLE,
                    MPI_SUM, world);
      bec_mean = bec_sum_global;
      if (natoms > 0) {
        for (int k = 0; k < 9; k++) bec_mean[k] /= static_cast<double>(natoms);
      }
    }

    for (int graph_idx = 0; graph_idx < nlocal; graph_idx++) {
      int i = graph_index_to_i[graph_idx];

      double c_xx = bec_cart[graph_idx][0] - bec_mean[0];
      double c_xy = bec_cart[graph_idx][1] - bec_mean[1];
      double c_xz = bec_cart[graph_idx][2] - bec_mean[2];
      double c_yx = bec_cart[graph_idx][3] - bec_mean[3];
      double c_yy = bec_cart[graph_idx][4] - bec_mean[4];
      double c_yz = bec_cart[graph_idx][5] - bec_mean[5];
      double c_zx = bec_cart[graph_idx][6] - bec_mean[6];
      double c_zy = bec_cart[graph_idx][7] - bec_mean[7];
      double c_zz = bec_cart[graph_idx][8] - bec_mean[8];

      double fx = (c_xx * efield[0] + c_xy * efield[1] + c_xz * efield[2]) * force->qe2f;
      double fy = (c_yx * efield[0] + c_yy * efield[1] + c_yz * efield[2]) * force->qe2f;
      double fz = (c_zx * efield[0] + c_zy * efield[1] + c_zz * efield[2]) * force->qe2f;

      tagint *tag = atom->tag;
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

      // Finite-field virial contribution (one-body external-field force,
      // not covered by the pairwise edge-vec (x) dE_dr virial below). Same
      // convention as LAMMPS's own fix_efield.cpp for a uniform field
      // acting on a per-atom force: virial_ab += F_a * r_b using
      // *unwrapped* atomic coordinates. This is the clamped-ion strain
      // derivative of the field enthalpy term -Omega*P.E; see enforce_asr
      // for its translation-invariance caveat under ASR violation.
      if (vflag) {
        double unwrap[3];
        domain->unmap(x[i], image[i], unwrap);
        virial[0] += fx * unwrap[0];
        virial[1] += fy * unwrap[1];
        virial[2] += fz * unwrap[2];
        virial[3] += fx * unwrap[1];
        virial[4] += fx * unwrap[2];
        virial[5] += fy * unwrap[2];
      }
    }
  }

  if (vflag) {
    auto diag = inp_edge_vec * dE_dr;
    auto s12 = inp_edge_vec.select(1, 0) * dE_dr.select(1, 1);
    auto s23 = inp_edge_vec.select(1, 1) * dE_dr.select(1, 2);
    auto s31 = inp_edge_vec.select(1, 2) * dE_dr.select(1, 0);
    std::vector<torch::Tensor> voigt_list = {
        diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)};
    auto voigt = torch::cat(voigt_list, 1);

    torch::Tensor per_atom_stress_tensor = torch::zeros({graph_indexer, 6});
    auto _edge_idx_dst6_tensor =
        edge_idx_dst_tensor.repeat_interleave(6).view({nedges, 6});
    per_atom_stress_tensor.scatter_reduce_(0, _edge_idx_dst6_tensor, voigt,
                                           "sum");
    auto virial_stress_tensor =
        torch::neg(torch::sum(per_atom_stress_tensor, 0));
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
    for (int graph_idx = 0; graph_idx < nlocal; graph_idx++) {
      int i = graph_index_to_i[graph_idx];
      eatom[i] += atomic_energy[graph_idx];
    }
  }

  // clean up comm preprocess variables
  comm_preprocess_done = false;
  for (int i = 0; i < 6; i++) {
    // array of vector<long>
    comm_index_pack_forward[i].clear();
    comm_index_unpack_forward[i].clear();
    comm_index_unpack_reverse[i].clear();
  }

  extra_graph_idx_map.clear();
}

// allocate arrays (called from coeff)
void PairE3GNNParallel::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(map, n + 1, "pair:map");
}

// global settings for pair_style
void PairE3GNNParallel::settings(int narg, char **arg) {
  if (narg != 0) {
    error->all(FLERR, "Illegal pair_style command");
  }
}

void PairE3GNNParallel::coeff(int narg, char **arg) {
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
      {"time", ""},
      {"flashTP", "version mismatch"},
      {"oeq", "version mismatch"},
      {"comm_size", ""}};

  // model loading from input
  int n_model = std::stoi(arg[2]);
  int chem_arg_i = 4;
  std::vector<std::string> model_fnames;
  if (std::filesystem::exists(arg[3])) {
    if (std::filesystem::is_directory(arg[3])) {
      auto headf = std::string(arg[3]);
      for (int i = 0; i < n_model; i++) {
        auto stri = std::to_string(i);
        model_fnames.push_back(headf + "/deployed_parallel_" + stri + ".pt");
      }
    } else if (std::filesystem::is_regular_file(arg[3])) {
      for (int i = 3; i < n_model + 3; i++) {
        model_fnames.push_back(std::string(arg[i]));
      }
      chem_arg_i = n_model + 3;
    } else {
      error->all(FLERR, "No such file or directory:" + std::string(arg[3]));
    }
  }

  for (const auto &modelf : model_fnames) {
    if (!std::filesystem::is_regular_file(modelf)) {
      error->all(FLERR, "Expected this is a regular file:" + modelf);
    }
    model_list.push_back(torch::jit::load(modelf, device, meta_dict));
  }

  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::FusionStrategy strategy;
  // strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  strategy = {{torch::jit::FusionBehavior::STATIC, 0}};
  torch::jit::setFusionStrategy(strategy);

  cutoff = std::stod(meta_dict["cutoff"]);

  // maximum possible size of per atom x before last convolution
  int comm_size = std::stod(meta_dict["comm_size"]);

  // to initialize buffer size for communication
  comm_forward = comm_size;
  comm_reverse = comm_size;

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

  // what if unknown chemical specie is in arg? should I abort? is there any use
  // case for that?
  if (narg > chem_arg_i && strcmp(arg[chem_arg_i], "efield") == 0) {
    if (narg < chem_arg_i + 4) {
      error->all(FLERR, "e3gnn/parallel: efield keyword requires 3 values (ex ey ez)");
    }
    has_efield = true;
    efield[0] = std::stod(arg[chem_arg_i + 1]);
    efield[1] = std::stod(arg[chem_arg_i + 2]);
    efield[2] = std::stod(arg[chem_arg_i + 3]);
    chem_arg_i += 4;

    if (narg > chem_arg_i && strcmp(arg[chem_arg_i], "enforce_asr") == 0) {
      enforce_asr = true;
      chem_arg_i += 1;
    }
  }

  bool found_flag = false;
  int n_chem = narg - chem_arg_i;
  for (int i = 0; i < n_chem; i++) {
    found_flag = false;
    for (int j = 0; j < chem_vec.size(); j++) {
      if (chem_vec[j].compare(arg[i + chem_arg_i]) == 0) {
        map[i + 1] = j; // store from 1, (not 0)
        found_flag = true;
        if (lmp->logfile) {
          fprintf(lmp->logfile, "Chemical specie '%s' is assigned to type %d\n",
                  arg[i + chem_arg_i], i + 1);
          break;
        }
      }
    }
    if (!found_flag) {
      error->all(FLERR, "Unknown chemical specie is given or the number of "
                        "potential files is not consistent");
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
void PairE3GNNParallel::init_style() {
  // full neighbor list & newton on
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style e3gnn/parallel requires newton pair on");
  }
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairE3GNNParallel::init_one(int i, int j) { return cutoff; }

void PairE3GNNParallel::notify_proc_ids(const int *sendproc, const int *recvproc) {
  for (int iswap = 0; iswap < 6; iswap++) {
    this->sendproc[iswap] = sendproc[iswap];
    this->recvproc[iswap]= recvproc[iswap];
  }
}

void PairE3GNNParallel::comm_preprocess() {
  assert(!comm_preprocess_done);
  CommBrick *comm_brick = dynamic_cast<CommBrick *>(comm);

  // fake lammps communication call to preprocess index
  // gives complete comm_index_pack, unpack_forward, and extra_graph_idx_map
  comm_brick->forward_comm(this);

  std::map<int, std::set<int>> already_met_map;
  for (int comm_phase = 0; comm_phase < 6; comm_phase++) {
    const int n = comm_index_pack_forward[comm_phase].size();
    int sproc = this->sendproc[comm_phase];
    if (already_met_map.count(sproc) == 0) {
      already_met_map.insert({sproc, std::set<int>()});
    }

    // for unpack_reverse, Ignore duplicated index by 'already_met'
    std::vector<long> &idx_map_forward = comm_index_pack_forward[comm_phase];
    std::vector<long> &idx_map_reverse = comm_index_unpack_reverse[comm_phase];
    std::set<int>& already_met = already_met_map[sproc];
    // the last index of x_comm is used to trash unnecessary values
    const int trash_index =
        graph_size + static_cast<int>(extra_graph_idx_map.size()); //+ 1;
    for (int i = 0; i < n; i++) {
      const int idx = idx_map_forward[i];
      if (idx < graph_size) {
        if (already_met.count(idx) == 1) {
          idx_map_reverse.push_back(trash_index);
        } else {
          idx_map_reverse.push_back(idx);
          already_met.insert(idx);
        }
      } else {
        idx_map_reverse.push_back(idx);
      }
    }

    if (use_cuda_mpi) {
      comm_index_pack_forward_tensor[comm_phase] = torch::from_blob(idx_map_forward.data(), idx_map_forward.size(), INTEGER_TYPE).to(device);

      auto upmap = comm_index_unpack_forward[comm_phase];
      comm_index_unpack_forward_tensor[comm_phase] = torch::from_blob(upmap.data(), upmap.size(), INTEGER_TYPE).to(device);
      comm_index_unpack_reverse_tensor[comm_phase] = torch::from_blob(idx_map_reverse.data(), idx_map_reverse.size(), INTEGER_TYPE).to(device);
    }
  }
  comm_preprocess_done = true;
}

// called from comm_brick if comm_preprocess_done is false
void PairE3GNNParallel::pack_forward_init(int n, int *list_send,
                                          int comm_phase) {
  std::vector<long> &idx_map = comm_index_pack_forward[comm_phase];

  idx_map.reserve(n);

  int i, j;
  int nlocal = list->inum;
  tagint *tag = atom->tag;

  for (i = 0; i < n; i++) {
    int list_i = list_send[i];
    int graph_idx = tag_to_graph_idx_ptr[tag[list_i]];

    if (graph_idx != -1) {
      // known atom (local atom + ghost atom inside cutoff)
      idx_map.push_back(graph_idx);
    } else {
      // unknown atom, these are not used in computation in this process
      // instead, this process is used to hand over these atoms to other proecss
      // hold them in continuous manner for flexible tensor operations later
      if (extra_graph_idx_map.find(list_i) != extra_graph_idx_map.end()) {
        idx_map.push_back(extra_graph_idx_map[list_i]);
      } else {
        // unknown atom at pack forward, ghost atom outside cutoff?
        extra_graph_idx_map[i] = graph_size + extra_graph_idx_map.size();
        idx_map.push_back(extra_graph_idx_map[i]); // same as list_i in pack
      }
    }
  }
}

// called from comm_brick if comm_preprocess_done is false
void PairE3GNNParallel::unpack_forward_init(int n, int first, int comm_phase) {
  std::vector<long> &idx_map = comm_index_unpack_forward[comm_phase];

  idx_map.reserve(n);

  int i, j, last;
  last = first + n;
  int nlocal = list->inum;
  tagint *tag = atom->tag;

  for (i = first; i < last; i++) {
    int graph_idx = tag_to_graph_idx_ptr[tag[i]];
    if (graph_idx != -1) {
      idx_map.push_back(graph_idx);
    } else {
      extra_graph_idx_map[i] = graph_size + extra_graph_idx_map.size();
      idx_map.push_back(extra_graph_idx_map[i]); // same as list_i in pack
    }
  }
}

int PairE3GNNParallel::pack_forward_comm_gnn(float *buf, int comm_phase) {
  std::vector<long> &idx_map = comm_index_pack_forward[comm_phase];
  const int n = static_cast<int>(idx_map.size());
  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_pack_forward_tensor[comm_phase];
    auto selected = x_comm.index_select(0, idx_map_tensor); // its size is x_dim * n
    cudaError_t cuda_err =
        cudaMemcpy(buf, selected.data_ptr<float>(), (x_dim * n) * sizeof(float),
                   cudaMemcpyDeviceToDevice);
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      float *from = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        buf[m++] = from[j];
      }
    }
  }
  return x_dim * n;
}

void PairE3GNNParallel::unpack_forward_comm_gnn(float *buf, int comm_phase) {
  std::vector<long> &idx_map = comm_index_unpack_forward[comm_phase];
  const int n = static_cast<int>(idx_map.size());

  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_unpack_forward_tensor[comm_phase];
    auto buf_tensor =
        torch::from_blob(buf, {n, x_dim}, FLOAT_TYPE.device(device));
    x_comm.scatter_(0, idx_map_tensor.repeat_interleave(x_dim).view({n, x_dim}),
                    buf_tensor);
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      float *to = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        to[j] = buf[m++];
      }
    }
  }
}

int PairE3GNNParallel::pack_reverse_comm_gnn(float *buf, int comm_phase) {
  std::vector<long> &idx_map = comm_index_unpack_forward[comm_phase];
  const int n = static_cast<int>(idx_map.size());

  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_unpack_forward_tensor[comm_phase];
    auto selected = x_comm.index_select(0, idx_map_tensor);
    cudaError_t cuda_err = cudaMemcpy(buf, selected.data_ptr<float>(), (x_dim * n) * sizeof(float), cudaMemcpyDeviceToDevice);
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      float *from = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        buf[m++] = from[j];
      }
    }
  }
  return x_dim * n;
}

void PairE3GNNParallel::unpack_reverse_comm_gnn(float *buf, int comm_phase) {
  std::vector<long> &idx_map = comm_index_unpack_reverse[comm_phase];
  const int n = static_cast<int>(idx_map.size());

  if (use_cuda_mpi && n != 0) {
    torch::Tensor &idx_map_tensor = comm_index_unpack_reverse_tensor[comm_phase];
    auto buf_tensor =
        torch::from_blob(buf, {n, x_dim}, FLOAT_TYPE.device(device));
    x_comm.scatter_(0, idx_map_tensor.repeat_interleave(x_dim).view({n, x_dim}),
                    buf_tensor, "add");
  } else {
    int i, j, m;
    m = 0;
    for (i = 0; i < n; i++) {
      const int idx = static_cast<int>(idx_map.at(i));
      if (idx == -1) {
        m += x_dim;
        continue;
      }
      float *to = x_comm[idx].data_ptr<float>();
      for (j = 0; j < x_dim; j++) {
        to[j] += buf[m++];
      }
    }
  }
}
