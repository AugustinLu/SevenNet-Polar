#!/bin/bash
# Build the Apple GPU pair style against `pip install lammps`. usage: build_plugin.sh [PYTHON]
# PYTHON is the environment with torch, ase, lammps and mpich: the plugin runs SevenNet inside it.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${1:-python}"
eval "$("$PY" - <<'PYEOF'
import datetime, os, shlex, sys, sysconfig, lammps
day = datetime.datetime.strptime(str(lammps.lammps(cmdargs=['-log', 'none', '-screen', 'none', '-nocite']).version()), '%Y%m%d')
v = dict(LMP=os.path.join(os.path.dirname(lammps.__file__), 'include', 'lammps'), PYINC=sysconfig.get_path('include'),
         PYLIB=sysconfig.get_config_var('LIBDIR'), PYVER=sysconfig.get_config_var('LDVERSION'), EXE=os.path.abspath(sys.executable), LMPVER=f'{day.day} {day:%b %Y}')
print('\n'.join(f'{k}={shlex.quote(x)}' for k, x in v.items()))
PYEOF
)"
clang++ -std=c++17 -O2 -fPIC -bundle -undefined dynamic_lookup -w \
  -I"$HERE" -I"$HERE/mpi_stub" -I"$LMP" -I"$PYINC" -DLAMMPS_SMALLBIG \
  -DSEVENN_DEFAULT_PYTHON="\"$EXE\"" -DSEVENN_DEFAULT_ROOT="\"$(cd "$HERE/../.." && pwd)\"" \
  "$HERE/pair_e3gnn_mps.cpp" "$HERE/e3gnn_mps_core.cpp" -o "$HERE/e3gnnmpscore.so"
clang++ -std=c++17 -O2 -fPIC -bundle -undefined dynamic_lookup -w \
  -DLAMMPS_VERSION="\"$LMPVER\"" -DSEVENN_LIBPYTHON="\"$PYLIB/libpython$PYVER.dylib\"" \
  "$HERE/e3gnn_mps_plugin.cpp" -o "$HERE/e3gnnmpsplugin.so"
echo "built $HERE/e3gnnmpsplugin.so"
echo "for lmp -in, set: export LAMMPS_PLUGIN_PATH=$HERE"
