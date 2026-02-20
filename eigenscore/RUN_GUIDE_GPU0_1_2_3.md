# GPU 0, 1, 2, 3 사용 가이드라인

## 📋 개요
OPT-6.7B 모델로 CoQA 데이터셋 전체(7,983개)를 GPU 0, 1, 2, 3에서 병렬로 처리합니다.
**레이어-가우시안 메서드**를 사용하며, **모든 레이어(0~32)**를 가우시안 가중 평균으로 계산합니다.

## 🚀 실행 단계

### 1단계: 사전 확인

```bash
# 프로젝트 디렉토리로 이동
cd /home/srjo/Gradproj/eigenscore

# GPU 상태 확인 (GPU 0, 1, 2, 3이 사용 가능한지 확인)
nvidia-smi

# Conda 환경 활성화
conda activate eigenscore
```

### 2단계: 실험 실행

```bash
# GPU 0, 1, 2, 3에서 병렬 실행
./run_gpu0_1_2_3.sh
```

**예상 소요 시간**: 약 45분~1시간 (각 GPU당 약 2,000개 데이터, 4개 GPU 병렬)

**데이터 분할**:
- GPU 0: 데이터 0~1994 (약 1,995개)
- GPU 1: 데이터 1995~3989 (약 1,995개)
- GPU 2: 데이터 3990~5984 (약 1,995개)
- GPU 3: 데이터 5985~7982 (약 1,998개)

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
kill <PID0> <PID1> <PID2> <PID3>
```

### 3단계: 결과 병합

실험이 완료되면 4개의 결과 파일을 병합합니다:

```bash
# 병합 스크립트 실행
python merge_gpu0_1_2_3.py
```

**병합 결과**:
- `output/opt-6.7b_coqa_gaussianlayer_merged/0.pkl` (약 7,983개)

### 4단계: 메트릭 계산

병합된 결과로 AUCs, AUCr, PCC를 계산합니다:

```bash
# evaluate_paper_metrics.py 실행
python evaluate_paper_metrics.py
```

또는 직접 실행:
```python
import pickle as pkl
from evaluate_paper_metrics import *

file_name = '/home/srjo/Gradproj/eigenscore/output/opt-6.7b_coqa_gaussianlayer_merged/0.pkl'

with open(file_name, 'rb') as f:
    resultDict = pkl.load(f)

print(f'총 {len(resultDict)}개 데이터 평가 시작')
# ... 평가 코드 실행
```

## 📁 결과 파일 위치

### 생성 파일:
- `output/opt-6.7b_coqa_gaussianlayer_0/0.pkl` - GPU 0 결과 (약 1,995개)
- `output/opt-6.7b_coqa_gaussianlayer_1/0.pkl` - GPU 1 결과 (약 1,995개)
- `output/opt-6.7b_coqa_gaussianlayer_2/0.pkl` - GPU 2 결과 (약 1,995개)
- `output/opt-6.7b_coqa_gaussianlayer_3/0.pkl` - GPU 3 결과 (약 1,998개)
- `output/opt-6.7b_coqa_gaussianlayer_merged/0.pkl` - 병합 결과 (약 7,983개)

### 로그 파일:
- `data/output/logInfo_opt-6.7b_coqa.txt`

## 🔬 실험 설정

### 레이어-가우시안 메서드
- **메서드**: `getEigenIndicator_layer_gaussian`
- **레이어 범위**: 모든 레이어 (0~32)
- **가중치**: 가우시안 분포 기반 가중 평균
  - μ (평균): 전체 레이어의 절반 (16)
  - σ (표준편차): 전체 레이어 수의 1/4 (8)
  - 가중치: w_l = (1/Z) * exp(-(l - μ)² / (2σ²))

### 파라미터
- **모델**: OPT-6.7B
- **데이터셋**: CoQA
- **Generation 수**: 10개 per prompt
- **Temperature**: 0.5
- **Top-p**: 0.99
- **Top-k**: 10

## ⚠️ 주의사항

1. **GPU 메모리**: OPT-6.7B는 각 GPU당 약 13GB 필요합니다
2. **디스크 공간**: 결과 파일은 약 20MB × 4 = 80MB 필요
3. **실행 중**: 다른 프로세스가 GPU 0, 1, 2, 3을 사용하지 않는지 확인
4. **레이어 설정**: 모든 레이어(0~32)를 사용하므로 계산 시간이 약간 더 걸릴 수 있습니다

## 🔍 문제 해결

### GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 다른 프로세스 종료
fuser -v /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3
kill <PID>
```

### 프로세스가 멈춤
```bash
# 로그 확인
tail -100 data/output/logInfo_opt-6.7b_coqa.txt

# 프로세스 상태 확인
ps aux | grep python
```

### 결과 파일이 없음
```bash
# 각 GPU별 결과 확인
ls -lh output/opt-6.7b_coqa_gaussianlayer_*/

# 특정 GPU가 실패했는지 확인
tail -50 data/output/logInfo_opt-6.7b_coqa.txt | grep -A 5 -B 5 "error\|Error\|ERROR"
```

## 📊 예상 결과

- **총 데이터**: 7,983개
- **처리 시간**: 약 45분~1시간 (4개 GPU 병렬)
- **최종 메트릭**: AUCs, AUCr, PCC (논문 Table 1 형식)

## 🔄 다른 메서드와 비교

### 레이어-평균 (averagelayer)
- 파일: `output/opt-6.7b_coqa_averagelayer_merged/0.pkl`
- 메서드: 단순 평균

### 레이어-가우시안 (gaussianlayer) ← **현재 실험**
- 파일: `output/opt-6.7b_coqa_gaussianlayer_merged/0.pkl`
- 메서드: 가우시안 가중 평균

## 📝 실행 예시

```bash
# 1. 실행
cd /home/srjo/Gradproj/eigenscore
conda activate eigenscore
./run_gpu0_1_2_3.sh

# 2. 병합 (실험 완료 후)
python merge_gpu0_1_2_3.py

# 3. 평가
python evaluate_paper_metrics.py
```
