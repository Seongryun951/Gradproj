import argparse
import os
import pickle as pkl
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import _settings
from func.metric import getRouge, rougeEvaluator
from func.plot import plot_weight_distribution


def load_data(pkl_file):
    """저장된 .pkl 파일에서 데이터 로드"""
    with open(pkl_file, 'rb') as f:
        resultDict = pkl.load(f)
    return resultDict


def extract_features_and_labels(resultDict, dataset_name):
    """
    resultDict에서 Feature Matrix X와 Label Vector y_rouge 추출
    
    Args:
        resultDict: 로드된 데이터 딕셔너리 리스트
        dataset_name: 데이터셋 이름 (coqa, nq_open 등)
    
    Returns:
        X: Feature Matrix (N x L) - 각 샘플의 레이어별 EigenScore 배열
        y_rouge: Label Vector (N,) - ROUGE-L > 0.5 기준
        sample_ids: 샘플 ID 리스트 (나중에 test set 필터링용)
    """
    X = []
    y_rouge = []
    sample_ids = []
    
    for item in resultDict:
        # layer_eigen_scores가 있는지 확인
        if 'layer_eigen_scores' not in item or len(item['layer_eigen_scores']) == 0:
            continue
        
        # Feature Matrix X 구성
        layer_scores = np.array(item['layer_eigen_scores'])
        X.append(layer_scores)
        
        # Label 생성 (ROUGE-L > 0.5 기준)
        ansGT = item['answer']
        generations = item['most_likely_generation']
        rougeScore = getRouge(rougeEvaluator, generations, ansGT)
        
        # CoQA나 TruthfulQA의 경우 additional_answers도 고려
        if dataset_name in ['coqa', 'TruthfulQA'] and 'additional_answers' in item:
            additional_answers = item['additional_answers']
            rougeScores = [getRouge(rougeEvaluator, generations, ans) for ans in additional_answers]
            rougeScore = max(rougeScore, max(rougeScores) if rougeScores else rougeScore)
        
        # Binary label: ROUGE-L > 0.5 -> 1, else -> 0
        label = 1 if rougeScore > 0.5 else 0
        y_rouge.append(label)
        
        # 샘플 ID 저장
        sample_ids.append(item['id'])
    
    X = np.array(X)
    y_rouge = np.array(y_rouge)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} layers")
    print(f"Label distribution: {np.sum(y_rouge)} positive, {len(y_rouge) - np.sum(y_rouge)} negative")
    
    return X, y_rouge, sample_ids


def train_logistic_regression(X_train, y_train, penalty='elasticnet', C=1.0, l1_ratio=0.5, max_iter=1000):
    """
    Logistic Regression 모델 학습
    
    Args:
        X_train: Training feature matrix (N x L)
        y_train: Training labels (N,)
        penalty: 'l1', 'l2', or 'elasticnet'
        C: Inverse of regularization strength
        l1_ratio: ElasticNet mixing parameter (0.5 = equal L1 and L2)
        max_iter: Maximum iterations
    
    Returns:
        model: 학습된 LogisticRegression 모델
        weights: 학습된 가중치 w = [W0, W1, ..., Wl]
    """
    solver = 'saga' if penalty in ['l1', 'elasticnet'] else 'lbfgs'
    
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 가중치 추출 (coef_[0]는 첫 번째 클래스의 가중치)
    weights = model.coef_[0]
    
    print(f"Trained Logistic Regression with {penalty} penalty")
    print(f"Number of non-zero weights: {np.sum(np.abs(weights) > 1e-6)} / {len(weights)}")
    
    return model, weights


def save_weights(weights, model_name, dataset_name, output_dir=None):
    """학습된 가중치를 파일로 저장"""
    if output_dir is None:
        output_dir = os.path.join(_settings._BASE_DIR, 'weights')
    os.makedirs(output_dir, exist_ok=True)
    
    # .pkl 파일로 저장
    weights_file = os.path.join(output_dir, f'{model_name}_{dataset_name}_weights.pkl')
    with open(weights_file, 'wb') as f:
        pkl.dump({
            'weights': weights,
            'model_name': model_name,
            'dataset_name': dataset_name
        }, f)
    
    # .json 파일로도 저장 (가독성)
    weights_json = os.path.join(output_dir, f'{model_name}_{dataset_name}_weights.json')
    with open(weights_json, 'w') as f:
        json.dump({
            'weights': weights.tolist(),
            'model_name': model_name,
            'dataset_name': dataset_name
        }, f, indent=2)
    
    print(f"Saved weights to {weights_file} and {weights_json}")
    return weights_file, weights_json


def main():
    parser = argparse.ArgumentParser(description='Train regression model for layer-wise EigenScore weights')
    parser.add_argument('--input', type=str, required=True, help='Input .pkl file path')
    parser.add_argument('--penalty', type=str, default='elasticnet', choices=['l1', 'l2', 'elasticnet'],
                        help='Regularization penalty type')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
    parser.add_argument('--l1_ratio', type=float, default=0.5, help='ElasticNet mixing parameter (0.5 = equal L1 and L2)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train/test split')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for weights (default: weights/)')
    
    args = parser.parse_args()
    
    # 데이터 로드
    print(f"Loading data from {args.input}...")
    resultDict = load_data(args.input)
    
    # 파일명에서 모델명과 데이터셋명 추출
    # 예: output/opt-6.7b_coqa_0/0.pkl -> opt-6.7b, coqa
    file_path = args.input
    path_parts = file_path.split('/')
    filename_parts = path_parts[-2].split('_') if len(path_parts) > 1 else path_parts[-1].split('_')
    model_name = filename_parts[0] if len(filename_parts) > 0 else 'unknown'
    dataset_name = filename_parts[1] if len(filename_parts) > 1 else 'unknown'
    
    # Feature와 Label 추출
    X, y_rouge, sample_ids = extract_features_and_labels(resultDict, dataset_name)
    
    if len(X) == 0:
        print("Error: No valid samples found with layer_eigen_scores")
        return
    
    # Train/Test Split
    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X, y_rouge, sample_ids,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_rouge  # 클래스 비율 유지
    )
    
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Test set 저장 (Phase 3에서 사용)
    test_data_file = args.input.replace('.pkl', '_test_data.pkl')
    test_data = {
        'X_test': X_test,
        'y_rouge_test': y_test,
        'test_ids': test_ids,
        'resultDict': [item for item in resultDict if item['id'] in test_ids]
    }
    with open(test_data_file, 'wb') as f:
        pkl.dump(test_data, f)
    print(f"Saved test set to {test_data_file}")
    
    # Logistic Regression 학습
    print("\nTraining Logistic Regression model...")
    model, weights = train_logistic_regression(
        X_train, y_train,
        penalty=args.penalty,
        C=args.C,
        l1_ratio=args.l1_ratio
    )
    
    # Train set 성능 평가
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    # Test set 성능 평가
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    # 가중치 저장
    weights_file, weights_json = save_weights(weights, model_name, dataset_name, args.output_dir)
    
    # 가중치 시각화
    plot_weight_distribution(weights, model_name, dataset_name)
    
    # 가중치 정보 출력
    print(f"\nLearned weights (first 10): {weights[:10]}")
    print(f"Weight statistics: min={np.min(weights):.4f}, max={np.max(weights):.4f}, mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
    
    # 모델 정보 저장
    model_info = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'penalty': args.penalty,
        'C': args.C,
        'l1_ratio': args.l1_ratio,
        'train_acc': float(train_acc),
        'train_auc': float(train_auc),
        'test_acc': float(test_acc),
        'test_auc': float(test_auc),
        'num_features': int(X.shape[1]),
        'num_train': int(len(X_train)),
        'num_test': int(len(X_test)),
        'test_data_file': test_data_file
    }
    
    info_file = weights_file.replace('.pkl', '_info.json')
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model info to {info_file}")


if __name__ == '__main__':
    main()
