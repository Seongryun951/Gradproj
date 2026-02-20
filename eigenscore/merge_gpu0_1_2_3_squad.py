#!/usr/bin/env python3
"""
GPU 0, 1, 2, 3에서 생성된 SQuAD pkl 파일을 병합하는 스크립트

사용법:
    python merge_gpu0_1_2_3_squad.py [project_ind]
    
예시:
    python merge_gpu0_1_2_3_squad.py                    # 기본값: 0 (getEigenIndicator_v0)
    python merge_gpu0_1_2_3_squad.py gaussianlayer      # gaussianlayer 사용
    python merge_gpu0_1_2_3_squad.py averagelayer       # averagelayer 사용
"""
import pickle as pkl
import os
import sys

# conda 환경 활성화
os.system("source ~/miniconda3/etc/profile.d/conda.sh && conda activate eigenscore")

def merge_gpu0_1_2_3_squad(project_ind=None):
    """
    GPU 0, 1, 2, 3의 SQuAD 결과를 병합합니다.
    
    파일명 형식: opt-6.7b_SQuAD_{project_ind}_{shard_id}/0.pkl
    - GPU 0: shard_id=0 → opt-6.7b_SQuAD_{project_ind}_0/0.pkl
    - GPU 1: shard_id=1 → opt-6.7b_SQuAD_{project_ind}_1/0.pkl
    - GPU 2: shard_id=2 → opt-6.7b_SQuAD_{project_ind}_2/0.pkl
    - GPU 3: shard_id=3 → opt-6.7b_SQuAD_{project_ind}_3/0.pkl
    
    Args:
        project_ind: 프로젝트 인덱스 (기본값: "0" - getEigenIndicator_v0 사용)
    """
    base_path = "/home/srjo/Gradproj/eigenscore/output"
    model_name = "opt-6.7b"
    dataset_name = "SQuAD"
    if project_ind is None:
        project_ind = "0"  # 기본값 (getEigenIndicator_v0 사용)
    
    merged_data = []
    
    # GPU 0, 1, 2, 3의 결과 로드 (shard_id 순서대로: 0, 1, 2, 3)
    for shard_id in [0, 1, 2, 3]:
        file_path = os.path.join(
            base_path, 
            f"{model_name}_{dataset_name}_{project_ind}_{shard_id}", 
            "0.pkl"
        )
        
        if not os.path.exists(file_path):
            print(f"경고: {file_path} 파일이 존재하지 않습니다.")
            continue
        
        print(f"로딩 중: {file_path}")
        with open(file_path, "rb") as f:
            data = pkl.load(f)
            merged_data.extend(data)
            print(f"  - {len(data)}개 항목 추가됨")
    
    print(f"\n총 {len(merged_data)}개 항목이 병합되었습니다.")
    
    # 병합된 파일 저장
    output_dir = os.path.join(base_path, f"{model_name}_{dataset_name}_{project_ind}_merged")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "0.pkl")
    
    with open(output_path, "wb") as f:
        pkl.dump(merged_data, f)
    
    print(f"병합 완료: {output_path}")
    return output_path

if __name__ == "__main__":
    # 명령줄 인자로 project_ind 받기
    project_ind = sys.argv[1] if len(sys.argv) > 1 else None
    merged_file = merge_gpu0_1_2_3_squad(project_ind=project_ind)
    print(f"\n병합된 파일: {merged_file}")
