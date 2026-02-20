#!/usr/bin/env python3
"""안정적인 모델 다운로드 스크립트"""

import os
from huggingface_hub import snapshot_download

# 환경 변수 설정
os.environ['HF_HUB_CACHE'] = '/havok_hdd/srjo/huggingface_cache'
os.environ.setdefault('HF_TOKEN', os.environ.get('HF_TOKEN', ''))

print("facebook/opt-6.7b 모델 다운로드 시작...")
print(f"캐시 디렉토리: {os.environ['HF_HUB_CACHE']}")

try:
    snapshot_download(
        repo_id="facebook/opt-6.7b",
        cache_dir="/havok_hdd/srjo/huggingface_cache",
        resume_download=True,
        max_workers=4,
    )
    print("\n✅ 다운로드 완료!")
except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    print("중단된 지점까지는 저장되었습니다. 다시 실행하면 이어서 받습니다.")
    exit(1)







    

