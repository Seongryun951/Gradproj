#!/usr/bin/env python3
"""
병합된 pkl 파일에 대해 메트릭(AUCs, AUCr, PCC)을 계산하는 스크립트
"""
import os
import sys
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_curve, auc
# func 모듈을 import하기 위해 경로 추가
sys.path.insert(0, '/home/srjo/Gradproj/eigenscore/func')
from metric import getRouge
from evalFunc import getPCC, getAUROC, getAcc, compute_exact_match
from rouge_score import rouge_scorer

if __name__ == "__main__":
    file_name = "/home/srjo/Gradproj/eigenscore/output/opt-6.7b_SQuAD_0_merged/0.pkl"
    print("="*60)
    print("병합된 데이터 평가 시작")
    print(f"파일: {file_name}")
    print("="*60)
    print()
    
    # 데이터 로드
    with open(file_name, "rb") as f:
        resultDict = pkl.load(f)
    
    print(f"총 데이터 개수: {len(resultDict)}개")
    print()
    
    # Accuracy 계산
    print("-"*60)
    print("1. Accuracy 계산")
    print("-"*60)
    getAcc(resultDict, file_name)
    print()
    

    # AUROC 및 PCC 계산
    
    
    
    
    print("-"*60)
    print("2. AUROC 및 PCC 계산")
    print("-"*60)
    getAUROC(resultDict, file_name)
    print()
    
    print("="*60)
    print("평가 완료!")
    print("="*60)

