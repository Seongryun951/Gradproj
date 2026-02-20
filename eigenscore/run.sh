export HF_HUB_CACHE=/havok_hdd/srjo/huggingface_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN 환경변수를 설정해주세요"}
python -m pipeline.generate --model opt-6.7b --dataset coqa --fraction_of_data_to_use 1 --project_ind 0
