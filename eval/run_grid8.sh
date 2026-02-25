set -euo pipefail

BASE_URL="http://127.0.0.1:8000"
DATASET="eval/eval_set.jsonl"
K=5

CHUNKS=("350" "800")
ALPHAS=("0.5" "0.8")
MINSCORES=("0.15" "0.25")

cd ~/ProdTraceRAG
mkdir -p reports

# server must be up
curl -s ${BASE_URL}/config >/dev/null

for chunk in "${CHUNKS[@]}"; do
  echo "== reset + ingest (chunk=${chunk}) =="
  curl -s -X POST ${BASE_URL}/reset_index >/dev/null

  overlap=$((chunk/6))
  curl -s -X POST ${BASE_URL}/ingest_local \
    -H 'Content-Type: application/json' \
    -d "{\"folder\":\"/path/to/ProdTraceRAG/demo_corpus\",\"glob_pattern\":\"**/*.md\",\"chunk_chars\":${chunk},\"overlap_chars\":${overlap}}" \
    | python -m json.tool | head -n 40

  echo "== config after ingest =="
  curl -s ${BASE_URL}/config | python -m json.tool | sed -n '1,120p'

  for alpha in "${ALPHAS[@]}"; do
    echo "== set_config alpha=${alpha} =="
    curl -s -X POST ${BASE_URL}/set_config \
      -H 'Content-Type: application/json' \
      -d "{\"alpha\": ${alpha}, \"min_evidence_score\": 0.08}" \
      | python -m json.tool

    for minscore in "${MINSCORES[@]}"; do
      RUN_NAME="grid8_chunk${chunk}_a${alpha}_ms${minscore}"
      echo "== eval: ${RUN_NAME} =="
      python eval/run_eval.py --base_url ${BASE_URL} --k ${K} --min_score ${minscore} --input ${DATASET} --run_name "${RUN_NAME}"
      echo ""
    done
  done
done

echo "DONE. Excel: reports/experiments.xlsx"
