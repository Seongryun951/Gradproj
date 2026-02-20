#!/bin/bash

export HF_HUB_CACHE=/havok_hdd/srjo/huggingface_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN 환경변수를 설정해주세요"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "  GPU 2, 3에서 병렬 실행 (OPT-6.7B, CoQA)"
echo "=========================================="
echo ""

# GPU 2 - 데이터의 1/2 (첫 번째 절반)
echo "[GPU 2] 첫 번째 1/2 데이터 처리 시작..."
CUDA_VISIBLE_DEVICES=2 python -m pipeline.generate \
    --model opt-6.7b \
    --dataset coqa \
    --fraction_of_data_to_use 1 \
    --project_ind averagelayer \
    --device cuda:0 \
    --shard_id 0 \
    --num_shards 2 &
PID2=$!

sleep 2

# GPU 3 - 데이터의 2/2 (두 번째 절반)
echo "[GPU 3] 두 번째 1/2 데이터 처리 시작..."
CUDA_VISIBLE_DEVICES=3 python -m pipeline.generate \
    --model opt-6.7b \
    --dataset coqa \
    --fraction_of_data_to_use 1 \
    --project_ind averagelayer \
    --device cuda:0 \
    --shard_id 1 \
    --num_shards 2 &
PID3=$!

echo ""
echo "=========================================="
echo "  작업 정보"
echo "=========================================="
echo "GPU 2 PID: $PID2 (데이터 0~3991)"
echo "GPU 3 PID: $PID3 (데이터 3992~7982)"
echo ""
echo "진행 상황 확인:"
echo "  - watch nvidia-smi"
echo "  - tail -f data/output/logInfo_opt-6.7b_coqa.txt"
echo ""
echo "작업 취소: kill $PID2 $PID3"
echo "=========================================="
echo ""

# 모든 프로세스가 끝날 때까지 대기
wait $PID2
echo "✓ GPU 2 작업 완료!"

wait $PID3
echo "✓ GPU 3 작업 완료!"

echo ""
echo "=========================================="
echo "  모든 GPU 작업 완료!"
echo "=========================================="
echo "결과 저장 위치:"
echo "  - output/opt-6.7b_coqa_averagelayer_0/0.pkl (약 3,992개)"
echo "  - output/opt-6.7b_coqa_averagelayer_1/0.pkl (약 3,991개)"
echo ""
echo "총 약 7,983개 데이터 처리 완료"
echo ""
echo "다음 단계:"
echo "  1. python merge_gpu2_3.py  # 결과 병합"
echo "  2. python evaluate_paper_metrics.py  # 메트릭 계산"
echo "=========================================="
