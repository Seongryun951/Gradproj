import argparse
import os
import pickle as pkl
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from func.metric import getRouge, rougeEvaluator, getSentenceSimilarity
from func.evalFunc import getPCC
from func.plot import VisAUROC
import _settings


def load_test_data(test_data_file):
    """Phase 2에서 저장한 test set 로드"""
    with open(test_data_file, 'rb') as f:
        test_data = pkl.load(f)
    return test_data


def load_weights(weights_file):
    """학습된 가중치 로드"""
    with open(weights_file, 'rb') as f:
        weights_data = pkl.load(f)
    return weights_data['weights']


def compute_semantic_similarity(resultDict, dataset_name):
    """
    SentenceTransformer를 사용하여 Semantic Similarity 계산
    
    Args:
        resultDict: 테스트 데이터 딕셔너리 리스트
        dataset_name: 데이터셋 이름
    
    Returns:
        similarities: 각 샘플의 semantic similarity 점수 리스트
    """
    # SentenceTransformer 모델 로드
    sentsim_local_path = './data/weights/nli-roberta-large'
    if os.path.exists(sentsim_local_path):
        SenSimModel = SentenceTransformer(sentsim_local_path)
    else:
        print("로컬에 nli-roberta-large 모델이 없습니다. Hugging Face에서 다운로드합니다...")
        SenSimModel = SentenceTransformer('sentence-transformers/nli-roberta-large')
    
    similarities = []
    for item in resultDict:
        ansGT = item['answer']
        generations = item['most_likely_generation']
        # getSentenceSimilarity는 문자열을 직접 받음 (evalFunc.py와 동일한 방식)
        similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
        
        # CoQA나 TruthfulQA의 경우 additional_answers도 고려
        if dataset_name in ['coqa', 'TruthfulQA'] and 'additional_answers' in item:
            additional_answers = item['additional_answers']
            similarities_list = [getSentenceSimilarity(generations, ans, SenSimModel) for ans in additional_answers]
            similarity = max(similarity, max(similarities_list) if similarities_list else similarity)
        
        similarities.append(similarity)
    
    return np.array(similarities)


def create_labels(resultDict, dataset_name, similarities=None):
    """
    두 가지 라벨 벡터 생성: y_rouge와 y_sim
    
    Args:
        resultDict: 테스트 데이터 딕셔너리 리스트
        dataset_name: 데이터셋 이름
        similarities: Semantic similarity 점수 배열 (None이면 계산)
    
    Returns:
        y_rouge: ROUGE-L > 0.5 기준 라벨
        y_sim: Semantic Similarity > 0.9 기준 라벨
        rouge_scores: ROUGE-L 점수 배열
        sim_scores: Semantic Similarity 점수 배열
    """
    y_rouge = []
    rouge_scores = []
    
    for item in resultDict:
        ansGT = item['answer']
        generations = item['most_likely_generation']
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        
        # CoQA나 TruthfulQA의 경우 additional_answers도 고려
        if dataset_name in ['coqa', 'TruthfulQA'] and 'additional_answers' in item:
            additional_answers = item['additional_answers']
            rougeScores = [getRouge(rougeEvaluator, generations, ans) for ans in additional_answers]
            rougeScore = max(rougeScore, max(rougeScores) if rougeScores else rougeScore)
        
        rouge_scores.append(rougeScore)
        y_rouge.append(1 if rougeScore > 0.5 else 0)
    
    y_rouge = np.array(y_rouge)
    rouge_scores = np.array(rouge_scores)
    
    # Semantic Similarity 라벨 생성
    if similarities is None:
        similarities = compute_semantic_similarity(resultDict, dataset_name)
    
    y_sim = (similarities > 0.9).astype(int)
    sim_scores = similarities
    
    return y_rouge, y_sim, rouge_scores, sim_scores


def compute_weighted_sum(X, weights):
    """
    Weighted Sum 계산: S(x) = Σ wl * El
    
    Args:
        X: Feature Matrix (N x L) - 각 샘플의 레이어별 EigenScore 배열
        weights: 학습된 가중치 [W0, W1, ..., Wl]
    
    Returns:
        weighted_scores: 각 샘플의 weighted sum 점수 (N,)
    """
    weighted_scores = np.dot(X, weights)
    return weighted_scores


def compute_baseline(X):
    """
    Baseline (Simple Average) 계산: S_baseline(x) = (1/L) * Σ El
    
    Args:
        X: Feature Matrix (N x L)
    
    Returns:
        baseline_scores: 각 샘플의 baseline 점수 (N,)
    """
    baseline_scores = np.mean(X, axis=1)
    return baseline_scores


def compute_auc(score, labels):
    """
    AUROC 계산 및 최적 threshold 계산
    
    Args:
        score: 예측 점수 배열
        labels: 실제 라벨 배열
    
    Returns:
        auc_score: AUROC 점수
        fpr: False Positive Rate
        tpr: True Positive Rate
        optimal_threshold: 최적 threshold (Youden's J statistic)
    """
    fpr, tpr, thresholds = roc_curve(labels, score)
    auc_score = auc(fpr, tpr)
    
    # 최적 threshold 계산 (Youden's J statistic: TPR * (1-FPR) 최대화)
    gmean = np.sqrt(tpr * (1 - fpr))
    optimal_idx = np.argmax(gmean)
    optimal_threshold = thresholds[optimal_idx]
    
    return auc_score, fpr, tpr, optimal_threshold


def main():
    parser = argparse.ArgumentParser(description='Evaluate regression model with weighted sum')
    parser.add_argument('--test_data', type=str, required=True, help='Test data .pkl file path (from Phase 2)')
    parser.add_argument('--weights', type=str, required=True, help='Learned weights .pkl file path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Test data 로드
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    X_test = test_data['X_test']
    resultDict = test_data['resultDict']
    
    # 가중치 로드
    print(f"Loading weights from {args.weights}...")
    weights = load_weights(args.weights)
    
    # 파일명에서 모델명과 데이터셋명 추출
    weights_path = args.weights
    path_parts = weights_path.split('/')
    filename = path_parts[-1] if len(path_parts) > 0 else weights_path
    filename_parts = filename.replace('_weights.pkl', '').split('_')
    model_name = filename_parts[0] if len(filename_parts) > 0 else 'unknown'
    dataset_name = filename_parts[1] if len(filename_parts) > 1 else 'unknown'
    
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"Test set size: {len(X_test)} samples, {X_test.shape[1]} layers")
    
    # 라벨 생성
    print("\nCreating labels...")
    similarities = compute_semantic_similarity(resultDict, dataset_name)
    y_rouge, y_sim, rouge_scores, sim_scores = create_labels(resultDict, dataset_name, similarities)
    
    print(f"y_rouge distribution: {np.sum(y_rouge)} positive, {len(y_rouge) - np.sum(y_rouge)} negative")
    print(f"y_sim distribution: {np.sum(y_sim)} positive, {len(y_sim) - np.sum(y_sim)} negative")
    
    # Weighted Sum 계산
    print("\nComputing weighted sum scores...")
    weighted_scores = compute_weighted_sum(X_test, weights)
    
    # Baseline 계산
    print("Computing baseline (simple average) scores...")
    baseline_scores = compute_baseline(X_test)
    
    # AUC_r 계산 (Weighted Sum vs y_rouge)
    print("\nEvaluating AUC_r (Weighted Sum vs y_rouge)...")
    auc_r_weighted, fpr_r_weighted, tpr_r_weighted, threshold_r_weighted = compute_auc(weighted_scores, y_rouge)
    auc_r_baseline, fpr_r_baseline, tpr_r_baseline, threshold_r_baseline = compute_auc(baseline_scores, y_rouge)
    
    # AUC_s 계산 (Weighted Sum vs y_sim)
    print("Evaluating AUC_s (Weighted Sum vs y_sim)...")
    auc_s_weighted, fpr_s_weighted, tpr_s_weighted, threshold_s_weighted = compute_auc(weighted_scores, y_sim)
    auc_s_baseline, fpr_s_baseline, tpr_s_baseline, threshold_s_baseline = compute_auc(baseline_scores, y_sim)
    
    # PCC 계산
    print("Computing Pearson Correlation Coefficients...")
    pcc_r_weighted = getPCC(weighted_scores, rouge_scores)
    pcc_r_baseline = getPCC(baseline_scores, rouge_scores)
    pcc_s_weighted = getPCC(weighted_scores, sim_scores)
    pcc_s_baseline = getPCC(baseline_scores, sim_scores)
    
    # 결과 출력
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nAUC_r (ROUGE-L based):")
    print(f"  Weighted Sum: AUC = {auc_r_weighted:.4f}, Threshold = {threshold_r_weighted:.4f}")
    print(f"    → S(x) >= {threshold_r_weighted:.4f}: 정답 (y=1)")
    print(f"    → S(x) < {threshold_r_weighted:.4f}: 할루시네이션 (y=0)")
    print(f"  Baseline: AUC = {auc_r_baseline:.4f}, Threshold = {threshold_r_baseline:.4f}")
    print(f"  Improvement: {auc_r_weighted - auc_r_baseline:.4f}")
    
    print(f"\nAUC_s (Semantic Similarity based):")
    print(f"  Weighted Sum: AUC = {auc_s_weighted:.4f}, Threshold = {threshold_s_weighted:.4f}")
    print(f"    → S(x) >= {threshold_s_weighted:.4f}: 정답 (y=1)")
    print(f"    → S(x) < {threshold_s_weighted:.4f}: 할루시네이션 (y=0)")
    print(f"  Baseline: AUC = {auc_s_baseline:.4f}, Threshold = {threshold_s_baseline:.4f}")
    print(f"  Improvement: {auc_s_weighted - auc_s_baseline:.4f}")
    
    print(f"\nPCC (Pearson Correlation Coefficient):")
    print(f"  Weighted Sum vs ROUGE-L: {pcc_r_weighted:.4f}")
    print(f"  Baseline vs ROUGE-L: {pcc_r_baseline:.4f}")
    print(f"  Weighted Sum vs Semantic Similarity: {pcc_s_weighted:.4f}")
    print(f"  Baseline vs Semantic Similarity: {pcc_s_baseline:.4f}")
    print("="*60)
    
    # 시각화
    if args.output_dir is None:
        output_dir = "./Figure"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # AUC_r curve 플롯
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_r_weighted, tpr_r_weighted, label=f"Weighted Sum (AUC={auc_r_weighted:.4f})", linewidth=2)
    plt.plot(fpr_r_baseline, tpr_r_baseline, label=f"Baseline (AUC={auc_r_baseline:.4f})", linewidth=2, linestyle='--')
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f'AUC_r: ROC Curve on {dataset_name} Dataset\n(ROUGE-L based)', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    auc_r_file = os.path.join(output_dir, f'AUC_r_{model_name}_{dataset_name}.png')
    plt.savefig(auc_r_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved AUC_r plot to {auc_r_file}")
    plt.close()
    
    # AUC_s curve 플롯
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_s_weighted, tpr_s_weighted, label=f"Weighted Sum (AUC={auc_s_weighted:.4f})", linewidth=2)
    plt.plot(fpr_s_baseline, tpr_s_baseline, label=f"Baseline (AUC={auc_s_baseline:.4f})", linewidth=2, linestyle='--')
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f'AUC_s: ROC Curve on {dataset_name} Dataset\n(Semantic Similarity based)', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    auc_s_file = os.path.join(output_dir, f'AUC_s_{model_name}_{dataset_name}.png')
    plt.savefig(auc_s_file, dpi=300, bbox_inches='tight')
    print(f"Saved AUC_s plot to {auc_s_file}")
    plt.close()
    
    # 결과 저장
    results = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'auc_r_weighted': float(auc_r_weighted),
        'auc_r_baseline': float(auc_r_baseline),
        'auc_s_weighted': float(auc_s_weighted),
        'auc_s_baseline': float(auc_s_baseline),
        'pcc_r_weighted': float(pcc_r_weighted),
        'pcc_r_baseline': float(pcc_r_baseline),
        'pcc_s_weighted': float(pcc_s_weighted),
        'pcc_s_baseline': float(pcc_s_baseline),
        'improvement_auc_r': float(auc_r_weighted - auc_r_baseline),
        'improvement_auc_s': float(auc_s_weighted - auc_s_baseline),
        'threshold_r_weighted': float(threshold_r_weighted),
        'threshold_r_baseline': float(threshold_r_baseline),
        'threshold_s_weighted': float(threshold_s_weighted),
        'threshold_s_baseline': float(threshold_s_baseline)
    }
    
    results_file = os.path.join(output_dir, f'evaluation_results_{model_name}_{dataset_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_file}")


if __name__ == '__main__':
    main()
