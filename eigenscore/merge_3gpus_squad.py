#!/usr/bin/env python3
"""
GPU 0, 1, 2에서 생성된 SQuAD pkl 파일을 병합하는 스크립트

사용법:
    python merge_3gpus_squad.py [project_ind]
    
예시:
    python merge_3gpus_squad.py                    # 기본값: 0
    python merge_3gpus_squad.py 1                  # project_ind=1 사용
    
주의: conda 환경이 활성화되어 있어야 합니다 (conda activate eigenscore)
"""
import pickle as pkl
import os
import sys

def merge_3gpus_squad(project_ind=None):
    """
    GPU 0, 1, 2의 SQuAD 결과를 병합합니다.
    
    파일명 형식: opt-6.7b_SQuAD_{project_ind}_{shard_id}/0.pkl
    - GPU 0: shard_id=0 → opt-6.7b_SQuAD_{project_ind}_0/0.pkl (데이터 1/3)
    - GPU 1: shard_id=1 → opt-6.7b_SQuAD_{project_ind}_1/0.pkl (데이터 2/3)
    - GPU 2: shard_id=2 → opt-6.7b_SQuAD_{project_ind}_2/0.pkl (데이터 3/3)
    
    Args:
        project_ind: 프로젝트 인덱스 (기본값: "0")
    """
    base_path = "/home/srjo/Gradproj/eigenscore/output"
    model_name = "opt-6.7b"
    dataset_name = "SQuAD"
    if project_ind is None:
        project_ind = "0"  # 기본값
    
    merged_data = []
    
    # GPU 0, 1, 2의 결과 로드 (shard_id 순서대로: 0, 1, 2)
    for shard_id in [0, 1, 2]:
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
    
    if len(merged_data) == 0:
        print("오류: 병합할 데이터가 없습니다.")
        return None
    
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
    merged_file = merge_3gpus_squad(project_ind=project_ind)
    if merged_file:
        print(f"\n병합된 파일: {merged_file}")
        print("\n다음 단계: 평가 메트릭 계산")
        print("  python func/evalFunc.py")
