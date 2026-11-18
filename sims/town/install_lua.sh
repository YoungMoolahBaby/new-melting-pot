#!/usr/bin/env bash
# Copy the clean_up_town Lua level into the Melting Pot package inside the venv.
# Melting Pot only loads levels from inside its own package
# (meltingpot/utils/substrates/builder.py: locate_and_overwrite_level_directory).
# Re-run after any change under sims/town/lua/.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${VENV:-$HOME/venvs/meltingpot}"
mp="$("$venv/bin/python" -I -c 'import meltingpot, os; print(os.path.dirname(meltingpot.__file__))')"
dest="$mp/lua/levels/clean_up_town"
mkdir -p "$dest"
cp "$here/lua/clean_up_town/"*.lua "$dest/"
echo "installed clean_up_town -> $dest"
