#!/usr/bin/env bash
# pack_gguf — produce the DEPLOYABLE GGUF artifact from the fake-quant clip checkpoint (CPU only).
# The framework's FINAL/deployment step: the fake-quant bf16 ckpt is the capability *ceiling* (what the eval
# harness measures); this GGUF pack is what actually RUNS on one GPU (or CPU + offload). Convert (text +
# vision mmproj) then quantize with the Tier-1 protected recipe (configs/gguf_tensor_types.txt): routed
# experts ~3b, shared-expert int8, attention int4, router/SSM f16, embed/lm_head int8.
#
#   bash pipelines/pack_gguf.sh
#   SRC=/models/q122_opd_r1 BASE_TYPE=IQ3_S bash pipelines/pack_gguf.sh
#
# Requires llama.cpp built with the qwen3_5_moe arch (PR #19468+, upstream 2026-02). On-device tok/s is the
# deployer's validation step (not run here — it is a benchmark, outside the packaging scope).
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck disable=SC1091
source "$HERE/env.sh"
: "${SRC:=models/q122_ptq3b_clip}"   # the fake-quant clip checkpoint (HF format)
: "${LLAMA_CPP:=./llama.cpp}"
: "${GGUF_OUT:=models/gguf}"
: "${BASE_TYPE:=Q3_K}"                                      # routed-expert bucket (~3.4bpw); IQ3_S to go lower (needs imatrix)
NAME="$(basename "$SRC")"
mkdir -p "$GGUF_OUT"
export PYTHONPATH="$LLAMA_CPP/gguf-py:${PYTHONPATH:-}"

# 1. convert. --no-mtp: the checkpoint has no mtp.* tensors despite config claiming mtp_num_hidden_layers=1.
[ -f "$GGUF_OUT/$NAME-bf16.gguf" ] || \
  python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" --outfile "$GGUF_OUT/$NAME-bf16.gguf" --outtype bf16 --no-mtp
# vision tower -> separate mmproj GGUF (use with llama-cli --mmproj for image input)
[ -f "$GGUF_OUT/$NAME-mmproj-f16.gguf" ] || \
  python "$LLAMA_CPP/convert_hf_to_gguf.py" "$SRC" --mmproj --outfile "$GGUF_OUT/$NAME-mmproj-f16.gguf" || \
  echo "[pack_gguf] mmproj convert skipped/failed (text GGUF still usable)"

# 2. quantize with the protected recipe (strip '#' comments — llama-quantize's parser does not skip them).
TT="$HERE/configs/gguf_tensor_types.txt"; CLEAN="$GGUF_OUT/.tt_clean.txt"
grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$TT" > "$CLEAN"
"$LLAMA_CPP/build/bin/llama-quantize" --tensor-type-file "$CLEAN" \
  --output-tensor-type q8_0 --token-embedding-type q8_0 \
  "$GGUF_OUT/$NAME-bf16.gguf" "$GGUF_OUT/$NAME-$BASE_TYPE-protected.gguf" "$BASE_TYPE"

echo "[pack_gguf] DONE -> $GGUF_OUT/$NAME-$BASE_TYPE-protected.gguf"
echo "[pack_gguf] deploy: $LLAMA_CPP/build/bin/llama-cli -m \"$GGUF_OUT/$NAME-$BASE_TYPE-protected.gguf\" \\"
echo "                    --mmproj \"$GGUF_OUT/$NAME-mmproj-f16.gguf\"   (on-device tok/s = deployer's validation)"
