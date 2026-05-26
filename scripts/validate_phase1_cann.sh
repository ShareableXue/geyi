#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-cann}"
if [[ "${TARGET}" != "cann" && "${TARGET}" != "scaffold" ]]; then
  echo "Usage: bash scripts/validate_phase1_cann.sh [cann|scaffold]" >&2
  exit 2
fi

PYTHON_CMD_TEXT="${GEYI_PYTHON:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_CMD_TEXT}"
BACKEND="${GEYI_BACKEND:-ascendc}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${GEYI_VALIDATE_OUT:-.geyi/phase1_validate_${TARGET}_${STAMP}}"
SUMMARY_FILE="${RUN_ROOT}/summary.tsv"

mkdir -p "${RUN_ROOT}/json" "${RUN_ROOT}/stderr" "${RUN_ROOT}/out"
printf "case\tstatus\tlevel\tmax_abs_diff\tartifact_hash\thardware\tjson\n" > "${SUMMARY_FILE}"

CASES=(
  "elementwise_mul|examples/elementwise_mul/mul.cu|examples/elementwise_mul/geyi.yaml"
  "elementwise_relu|examples/elementwise_relu/relu.cu|examples/elementwise_relu/geyi.yaml"
  "elementwise_neg|examples/elementwise_neg/neg.cu|examples/elementwise_neg/geyi.yaml"
  "copy1d|examples/copy1d/copy.cu|examples/copy1d/geyi.yaml"
  "cast1d|examples/cast1d/cast.cu|examples/cast1d/geyi.yaml"
)

if [[ "${GEYI_INCLUDE_VECTOR_ADD:-0}" == "1" ]]; then
  CASES+=("vector_add|examples/vector_add/vector_add.cu|examples/vector_add/geyi.yaml")
fi

if [[ "${GEYI_INCLUDE_2D:-0}" == "1" ]]; then
  CASES+=(
    "transpose2d|examples/transpose2d/transpose.cu|examples/transpose2d/geyi.yaml"
    "row_reduce_sum|examples/row_reduce_sum/row_sum.cu|examples/row_reduce_sum/geyi.yaml"
  )
fi

echo "Geyi Phase 1 validation"
echo "  backend: ${BACKEND}"
echo "  target:  ${TARGET}"
echo "  python:  ${PYTHON_CMD_TEXT}"
echo "  out:     ${RUN_ROOT}"
echo

failures=0
for record in "${CASES[@]}"; do
  IFS="|" read -r name source spec <<< "${record}"
  json_file="${RUN_ROOT}/json/${name}.json"
  stderr_file="${RUN_ROOT}/stderr/${name}.stderr.log"
  out_dir="${RUN_ROOT}/out/${name}"

  echo "==> ${name}"
  cmd=(
    "${PYTHON_CMD[@]}" -m geyi.cli.main run "${source}"
    --spec "${spec}"
    --backend "${BACKEND}"
    --target "${TARGET}"
    --out "${out_dir}"
    --json
  )

  if "${cmd[@]}" > "${json_file}" 2> "${stderr_file}"; then
    if summary_line="$("${PYTHON_CMD[@]}" -c '
import json
import sys

json_file, target, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(json_file, "r", encoding="utf-8") as f:
    payload = json.load(f)

if not payload.get("passed"):
    raise SystemExit("%s reported passed=False" % name)

hardware = payload.get("coverage", {}).get("hardware", [])
if target == "cann" and "ascend_npu" not in hardware:
    raise SystemExit("%s did not claim ascend_npu hardware: %r" % (name, hardware))
if target == "scaffold" and "not_run" not in hardware:
    raise SystemExit("%s scaffold did not report not_run hardware: %r" % (name, hardware))

fields = [
    name,
    "PASS",
    str(payload.get("level")),
    str(payload.get("max_abs_diff")),
    str(payload.get("artifact_hash")),
    ",".join(hardware),
    json_file,
]
print("\t".join(fields))
' "${json_file}" "${TARGET}" "${name}")"; then
      echo "${summary_line}" | tee -a "${SUMMARY_FILE}"
    else
      failures=$((failures + 1))
      printf "%s\tFAIL\tparse\t\t\t\t%s\n" "${name}" "${json_file}" | tee -a "${SUMMARY_FILE}"
      echo "JSON validation failed for ${name}" >&2
      tail -n 80 "${json_file}" >&2 || true
    fi
  else
    failures=$((failures + 1))
    printf "%s\tFAIL\tcommand\t\t\t\t%s\n" "${name}" "${json_file}" | tee -a "${SUMMARY_FILE}"
    echo "Command failed for ${name}" >&2
    echo "--- stderr tail ---" >&2
    tail -n 80 "${stderr_file}" >&2 || true
    echo "--- stdout/json tail ---" >&2
    tail -n 80 "${json_file}" >&2 || true
  fi
  echo
done

echo "Summary: ${SUMMARY_FILE}"
if [[ "${failures}" -ne 0 ]]; then
  echo "Validation failed: ${failures} case(s)" >&2
  exit 1
fi

echo "Validation passed"
