#!/usr/bin/env python3
"""
논문 Table 1 재현: AUCs, AUCr, PCC 계산
AUCs: Sentence Similarity 기준 AUROC
AUCr: ROUGE-L 기준 AUROC
PCC: Pearson Correlation Coefficient
"""
import os
import argparse

# HuggingFace 캐시 경로 설정 (모델 다운로드 방지)
os.environ['HF_HUB_CACHE'] = '/havok_hdd/srjo/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/havok_hdd/srjo/huggingface_cache'
import sys
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_curve, auc
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# func 모듈 import
sys.path.insert(0, '/home/srjo/Gradproj/eigenscore/func')
from metric import getRouge, getSentenceSimilarity

# ROUGE 평가기 초기화
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def normalize_text(s):
    """텍스트 정규화"""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def getPCC(x, y):
    """Pearson Correlation Coefficient 계산"""
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]

def evaluate_with_correctness_measure(resultDict, file_name, use_sentence_sim=False, SenSimModel=None):
    """
    특정 correctness measure로 평가
    use_sentence_sim=True: AUCs (Sentence Similarity)
    use_sentence_sim=False: AUCr (ROUGE-L)
    """
    Label = []
    Score = []
    Perplexity = []
    Energy = []
    LexicalSimilarity = []
    SentBertScore = []
    Entropy = []
    EigenIndicator = []
    EigenIndicatorOutput = []
    
    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        
        # 메트릭 수집
        Perplexity.append(-item["perplexity"])
        Energy.append(-item["energy"])
        Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        SentBertScore.append(-item["sent_bertscore"])
        EigenIndicator.append(-item["eigenIndicator"])
        EigenIndicatorOutput.append(-item["eigenIndicatorOutput"])
        
        # Correctness measure 선택
        if use_sentence_sim:
            # AUCs: Sentence Similarity
            similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
            if "coqa" in file_name.lower() or "truthfulqa" in file_name.lower():
                additional_answers = item["additional_answers"]
                similarities = [getSentenceSimilarity(generations, ans, SenSimModel) for ans in additional_answers]
                similarity = max(similarity, max(similarities))
            
            if similarity > 0.9:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        else:
            # AUCr: ROUGE-L
            rougeScore = getRouge(rougeEvaluator, generations, ansGT)
            if "coqa" in file_name.lower() or "truthfulqa" in file_name.lower():
                additional_answers = item["additional_answers"]
                rougeScores = [getRouge(rougeEvaluator, generations, ans) for ans in additional_answers]
                rougeScore = max(rougeScore, max(rougeScores))
            
            if rougeScore > 0.5:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(rougeScore)
    
    # AUROC 계산
    results = {}
    
    metrics = {
        'Perplexity': Perplexity,
        'Energy': Energy,
        'Entropy': Entropy,
        'LexicalSimilarity': LexicalSimilarity,
        'SentBertScore': SentBertScore,
        'EigenScore': EigenIndicator,
        'EigenScore-Output': EigenIndicatorOutput
    }
    
    for name, metric_values in metrics.items():
        fpr, tpr, thresholds = roc_curve(Label, metric_values)
        auroc = auc(fpr, tpr)
        pcc = getPCC(Score, metric_values)
        results[name] = {'AUROC': auroc, 'PCC': pcc}
    
    return results

def print_results_table(results_s, results_r, method_name="", dataset="SQuAD"):
    """결과를 표 형식으로 출력 (논문 Table 형식)"""
    print("\n" + "="*90)
    print(f"논문 Table 형식 결과 (OPT-6.7B, {dataset})")
    if method_name:
        print(f"Method: {method_name}")
    print("="*90)
    
    # 메서드 순서 정의 (논문 표 형식에 맞춤 - 이미지 참고)
    # 이미지에는 5개 행이 있고, 마지막 행이 EigenScore-Output (bold, 가장 높은 값)
    method_order = ['Perplexity', 'Energy', 'Entropy', 'LexicalSimilarity', 'EigenScore-Output']
    
    # 메서드 이름 매핑 (짧은 이름으로 표시)
    method_names = {
        'Perplexity': 'Perplexity',
        'Energy': 'Energy',
        'Entropy': 'Entropy',
        'LexicalSimilarity': 'LexicalSim',
        'EigenScore-Output': 'EigenScore'
    }
    
    # 표 헤더 (이미지 형식에 맞춤: C | AUCs | AUCr | PCC | A)
    print(f"{'Method':<15} | {'AUCs':<8} | {'AUCr':<8} | {'PCC':<8}")
    print("-"*90)
    
    for idx, method in enumerate(method_order, 1):
        if method in results_s:
            auc_s = results_s[method]['AUROC'] * 100
            auc_r = results_r[method]['AUROC'] * 100
            pcc_val = results_s[method]['PCC']
            # NaN 처리
            if np.isnan(pcc_val):
                pcc_str = "  nan"
            else:
                pcc_str = f"{pcc_val * 100:>7.1f}"
            
            # 메서드 이름 가져오기
            method_name = method_names.get(method, method)
            
            # 마지막 행 (EigenScore-Output)은 bold 표시를 위해 별도 처리
            if method == 'EigenScore-Output':
                # 이미지처럼 마지막 행은 bold (터미널에서는 **로 표시하거나 그냥 출력)
                print(f"{method_name:<15} | {auc_s:>7.1f} | {auc_r:>7.1f} | {pcc_str}")
            else:
                print(f"{method_name:<15} | {auc_s:>7.1f} | {auc_r:>7.1f} | {pcc_str}")
    
    # EigenScore도 참고용으로 표시 (이미지에는 없지만 비교용)
    if 'EigenScore' in results_s and 'EigenScore-Output' not in method_order:
        auc_s_eigen = results_s['EigenScore']['AUROC'] * 100
        auc_r_eigen = results_r['EigenScore']['AUROC'] * 100
        pcc_eigen = results_s['EigenScore']['PCC']
        if np.isnan(pcc_eigen):
            pcc_str_eigen = "  nan"
        else:
            pcc_str_eigen = f"{pcc_eigen * 100:>7.1f}"
        print("-"*90)
        print(f"{'Eigen':<5} | {auc_s_eigen:>7.1f} | {auc_r_eigen:>7.1f} | {pcc_str_eigen} | {'':<5}")
    
    print("="*90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='논문 Table 1 재현: AUCs, AUCr, PCC 측정')
    parser.add_argument('--file', type=str, 
                       default="/home/srjo/Gradproj/eigenscore/output/opt-6.7b_SQuAD_0_merged/0.pkl",
                       help='평가할 pkl 파일 경로')
    parser.add_argument('--compare', action='store_true',
                       help='averagelayer와 gaussianlayer 결과를 비교')
    args = parser.parse_args()
    
    # 비교 모드
    if args.compare:
        base_path = "/home/srjo/Gradproj/eigenscore/output"
        files = {
            'averagelayer': f"{base_path}/opt-6.7b_SQuAD_averagelayer_merged/0.pkl",
            'gaussianlayer': f"{base_path}/opt-6.7b_SQuAD_gaussianlayer_merged/0.pkl"
        }
        
        # Sentence Similarity 모델 로드 (한 번만)
        print("\nSentence Similarity 모델 로딩 중...")
        SenSimModel = SentenceTransformer('sentence-transformers/nli-roberta-large')
        print("모델 로드 완료 (HuggingFace 캐시 사용)")
        
        all_results = {}
        
        for method_name, file_name in files.items():
            if not os.path.exists(file_name):
                print(f"\n⚠️  경고: {file_name} 파일이 존재하지 않습니다. 건너뜁니다.")
                continue
                
            print("\n" + "="*90)
            print(f"{method_name.upper()} 메서드 평가")
            print("="*90)
            print(f"파일: {file_name}")
            
            # 데이터 로드
            print("\n데이터 로딩 중...")
            with open(file_name, "rb") as f:
                resultDict = pkl.load(f)
            print(f"총 {len(resultDict)}개 데이터 로드 완료")
            
            # AUCr 계산 (ROUGE-L 기준)
            print("\n" + "-"*90)
            print("1. AUCr 계산 중 (ROUGE-L 기준)...")
            print("-"*90)
            results_r = evaluate_with_correctness_measure(resultDict, file_name, use_sentence_sim=False)
            print("✓ AUCr 계산 완료")
            
            # AUCs 계산 (Sentence Similarity 기준)
            print("\n" + "-"*90)
            print("2. AUCs 계산 중 (Sentence Similarity 기준)...")
            print("-"*90)
            results_s = evaluate_with_correctness_measure(resultDict, file_name, use_sentence_sim=True, SenSimModel=SenSimModel)
            print("✓ AUCs 계산 완료")
            
            # 결과 저장
            all_results[method_name] = {'results_s': results_s, 'results_r': results_r}
            
        # 결과 출력
        print_results_table(results_s, results_r, method_name, dataset="SQuAD")
        
        # 비교 결과 출력
        if len(all_results) == 2:
            print("\n" + "="*90)
            print("메서드 비교 결과 (EigenScore만)")
            print("="*90)
            print(f"{'Method':<20} | {'AUC_s':<8} | {'AUC_r':<8} | {'PCC':<8}")
            print("-"*90)
            for method_name, results in all_results.items():
                eigen_s = results['results_s']['EigenScore']
                eigen_r = results['results_r']['EigenScore']
                print(f"{method_name:<20} | {eigen_s['AUROC']*100:>7.1f} | {eigen_r['AUROC']*100:>7.1f} | {eigen_s['PCC']*100:>7.1f}")
            print("="*90)
    
    else:
        # 단일 파일 평가
        file_name = args.file
        
        print("="*90)
        print("논문 Table 재현: AUCs, AUCr, PCC 측정 (SQuAD)")
        print("="*90)
        print(f"파일: {file_name}")
        
        # 데이터 로드
        print("\n데이터 로딩 중...")
        with open(file_name, "rb") as f:
            resultDict = pkl.load(f)
        print(f"총 {len(resultDict)}개 데이터 로드 완료")
        
        # Sentence Similarity 모델 로드 (AUCs용)
        print("\nSentence Similarity 모델 로딩 중...")
        SenSimModel = SentenceTransformer('sentence-transformers/nli-roberta-large')
        print("모델 로드 완료 (HuggingFace 캐시 사용)")
        
        # AUCr 계산 (ROUGE-L 기준)
        print("\n" + "-"*90)
        print("1. AUCr 계산 중 (ROUGE-L 기준)...")
        print("-"*90)
        results_r = evaluate_with_correctness_measure(resultDict, file_name, use_sentence_sim=False)
        print("✓ AUCr 계산 완료")
        
        # AUCs 계산 (Sentence Similarity 기준)
        print("\n" + "-"*90)
        print("2. AUCs 계산 중 (Sentence Similarity 기준)...")
        print("-"*90)
        results_s = evaluate_with_correctness_measure(resultDict, file_name, use_sentence_sim=True, SenSimModel=SenSimModel)
        print("✓ AUCs 계산 완료")
        
        # 결과 출력
        # 파일명에서 데이터셋 추출
        dataset = "SQuAD" if "SQuAD" in file_name else "CoQA"
        print_results_table(results_s, results_r, dataset=dataset)
    
    print("\n" + "="*90)
    print("측정 완료!")
    print("="*90)

