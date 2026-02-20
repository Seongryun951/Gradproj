# GPU 2, 3 사용 가이드라인

## 📋 개요
OPT-6.7B 모델로 CoQA 데이터셋 전체(7,983개)를 GPU 2, 3에서 병렬로 처리합니다.

## 🚀 실행 단계

### 1단계: 사전 확인

```bash
# 프로젝트 디렉토리로 이동
cd /home/srjo/Gradproj/eigenscore

# GPU 상태 확인 (GPU 2, 3이 사용 가능한지 확인)
nvidia-smi

# Conda 환경 활성화
conda activate eigenscore
```

### 2단계: 실험 실행

```bash
# GPU 2, 3에서 병렬 실행
./run_gpu2_3.sh
```

**예상 소요 시간**: 약 1.5시간 (각 GPU당 약 3,992개 데이터)

**실행 중 확인 방법**:
```bash
# GPU 사용률 확인
watch nvidia-smi

# 로그 확인
tail -f data/output/logInfo_opt-6.7b_coqa.txt
```

**실행 중단 방법**:
```bash
# PID 확인 후 종료
ps aux | grep pipeline.generate
kill <PID2> <PID3>
```

### 3단계: 결과 병합

실험이 완료되면 2개의 결과 파일을 병합합니다:

```bash
# 병합 스크립트 실행 (num_shards=2로 수정 필요)
python -c "
import pickle as pkl
import os

base_path = '/home/srjo/Gradproj/eigenscore/output'
model_name = 'opt-6.7b'
dataset_name = 'coqa'

merged_data = []
for shard_id in [2, 3]:  # GPU 2, 3의 결과
    file_path = os.path.join(base_path, f'{model_name}_{dataset_name}_{shard_id}', '0.pkl')
    print(f'로딩 중: {file_path}')
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
        merged_data.extend(data)
        print(f'  - {len(data)}개 항목 추가됨')

print(f'\n총 {len(merged_data)}개 항목이 병합되었습니다.')

output_dir = os.path.join(base_path, f'{model_name}_{dataset_name}_merged_gpu23')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '0.pkl')

with open(output_path, 'wb') as f:
    pkl.dump(merged_data, f)

print(f'병합 완료: {output_path}')
"
```

### 4단계: 메트릭 계산

병합된 결과로 AUCs, AUCr, PCC를 계산합니다:

```bash
# evaluate_paper_metrics.py의 파일 경로를 수정해야 합니다
# 또는 직접 실행:
python -c "
import sys
sys.path.insert(0, '/home/srjo/Gradproj/eigenscore')
from evaluate_paper_metrics import *

# 파일 경로 수정
file_name = '/home/srjo/Gradproj/eigenscore/output/opt-6.7b_coqa_merged_gpu23/0.pkl'

# 데이터 로드 및 평가
with open(file_name, 'rb') as f:
    resultDict = pkl.load(f)

print(f'총 {len(resultDict)}개 데이터 평가 시작')
# ... 평가 코드 실행
"
```

## 📁 결과 파일 위치

### 생성 파일:
- `output/opt-6.7b_coqa_2/0.pkl` - GPU 2 결과 (약 3,992개)
- `output/opt-6.7b_coqa_3/0.pkl` - GPU 3 결과 (약 3,991개)
- `output/opt-6.7b_coqa_merged_gpu23/0.pkl` - 병합 결과 (약 7,983개)

### 로그 파일:
- `data/output/logInfo_opt-6.7b_coqa.txt`

## ⚠️ 주의사항

1. **GPU 메모리**: OPT-6.7B는 각 GPU당 약 13GB 필요합니다
2. **디스크 공간**: 결과 파일은 약 80MB × 2 = 160MB 필요
3. **실행 중**: 다른 프로세스가 GPU 2, 3을 사용하지 않는지 확인

## 🔍 문제 해결

### GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 다른 프로세스 종료
fuser -v /dev/nvidia2 /dev/nvidia3
kill <PID>
```

### 프로세스가 멈춤
```bash
# 로그 확인
tail -100 data/output/logInfo_opt-6.7b_coqa.txt

# 프로세스 상태 확인
ps aux | grep python
```

## 📊 예상 결과

- **총 데이터**: 7,983개
- **처리 시간**: 약 1.5시간
- **최종 메트릭**: AUCs, AUCr, PCC (논문 Table 1 형식)
