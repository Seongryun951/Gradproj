#!/usr/bin/env python3
"""
4개의 GPU에서 생성된 pkl 파일을 병합하는 스크립트
"""
import pickle as pkl
import os

def merge_pkl_files(base_path, model_name, dataset_name, num_shards):
    """
    여러 shard로 나뉜 pkl 파일들을 병합합니다.
    
    Args:
        base_path: output 디렉토리 경로
        model_name: 모델 이름 (예: opt-6.7b)
        dataset_name: 데이터셋 이름 (예: coqa)
        num_shards: shard 개수
    """
    merged_data = []
    
    for shard_id in range(num_shards):
        file_path = os.path.join(
            base_path, 
            f"{model_name}_{dataset_name}_{shard_id}", 
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
    output_dir = os.path.join(base_path, f"{model_name}_{dataset_name}_merged")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "0.pkl")
    
    with open(output_path, "wb") as f:
        pkl.dump(merged_data, f)
    
    print(f"병합 완료: {output_path}")
    return output_path

if __name__ == "__main__":
    base_path = "/home/srjo/Gradproj/eigenscore/output"
    model_name = "opt-6.7b"
    dataset_name = "coqa"
    num_shards = 4
    
    merged_file = merge_pkl_files(base_path, model_name, dataset_name, num_shards)
    print(f"\n병합된 파일: {merged_file}")

