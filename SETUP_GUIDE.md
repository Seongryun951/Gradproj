# EigenScore 환경 설정 가이드

## 📋 개요
eigenscore를 실행하기 위해서는 Python 3.10과 여러 패키지가 필요합니다.
아래 단계를 따라 환경을 설정하세요.

---

## 🚀 설치 단계

### 1단계: Miniconda 설치

```bash
cd /home/srjo/Gradproj
bash setup_conda.sh
```

설치가 완료되면 다음 명령어를 실행하세요:

```bash
source ~/.bashrc
```

### 2단계: Conda 환경 생성

```bash
cd /home/srjo/Gradproj
conda env create -f eigenscore.yml
```

이 과정은 몇 분 정도 걸릴 수 있습니다.

### 3단계: 환경 활성화

```bash
conda activate eigenscore
```

### 4단계: eigenscore 실행

```bash
cd /home/srjo/Gradproj/eigenscore
```

이제 eigenscore를 실행할 수 있습니다!

---

## 📝 설치 확인

환경이 제대로 설치되었는지 확인하려면:

```bash
conda activate eigenscore
python --version  # Python 3.10.x가 나와야 합니다
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🔄 다음에 사용할 때

시스템을 재시작하거나 새 터미널을 열었을 때:

```bash
conda activate eigenscore
cd /home/srjo/Gradproj/eigenscore
# 작업 시작
```

---

## ❓ 문제 해결

### conda 명령어가 작동하지 않는 경우:

```bash
source ~/.bashrc
```

또는

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

### 환경 목록 확인:

```bash
conda env list
```

### 환경 삭제 (재설치가 필요한 경우):

```bash
conda env remove -n eigenscore
conda env create -f eigenscore.yml
```

---

## 📦 설치되는 주요 패키지

- Python 3.10
- PyTorch (GPU 지원)
- Transformers (Hugging Face)
- OpenAI API
- scikit-learn
- pandas, numpy
- 그 외 eigenscore 실행에 필요한 패키지들

