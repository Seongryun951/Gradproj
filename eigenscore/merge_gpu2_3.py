#!/usr/bin/env python3
"""
GPU 2, 3에서 생성된 pkl 파일을 병합하는 스크립트
"""
import pickle as pkl
import os

def merge_gpu2_3():
    """
    GPU 2, 3의 결과를 병합합니다.
    
    파일명 형식: opt-6.7b_coqa_averagelayer_{shard_id}/0.pkl
    - GPU 2: shard_id=0 → opt-6.7b_coqa_averagelayer_0/0.pkl (데이터 0~3990)
    - GPU 3: shard_id=1 → opt-6.7b_coqa_averagelayer_1/0.pkl (데이터 3991~7982)
    """
    base_path = "/home/srjo/Gradproj/eigenscore/output"
    model_name = "opt-6.7b"
    dataset_name = "coqa"
    project_ind = "averagelayer"
    
    merged_data = []
    
    # GPU 2, 3의 결과 로드 (shard_id 순서대로: 0, 1)
    # 이 순서가 중요: GPU 2가 shard_id=0이므로 먼저 와야 함
    for shard_id in [0, 1]:
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
    merged_file = merge_gpu2_3()
    print(f"\n병합된 파일: {merged_file}")
