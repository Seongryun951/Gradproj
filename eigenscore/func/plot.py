import matplotlib.pyplot as plt
import numpy as np
import os



def VisAUROC(tpr, fpr, AUROC, method_name, file_name="CoQA"):
    if "coqa" in file_name:
        file_name = "CoQA"
    if "nq" in file_name:
        file_name = "NQ"
    if "trivia" in file_name:
        file_name = "TriviaQA"
    if "SQuAD" in file_name:
        file_name = "SQuAD"
    plt.plot(fpr, tpr, label="AUC-{}=".format(method_name)+str(round(AUROC,3)))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('ROC Curve on {} Dataset'.format(file_name), fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig("./Figure/AUROC_{}.png".format(file_name), dpi=300, bbox_inches='tight')
    plt.show()



def plot_weight_distribution(weights, model_name, dataset_name, output_dir=None):
    """
    학습된 레이어별 가중치 분포를 bar chart로 시각화
    
    Args:
        weights: 학습된 가중치 배열 [W0, W1, ..., Wl]
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        output_dir: 출력 디렉토리 (None이면 ./Figure/)
    """
    if output_dir is None:
        output_dir = "./Figure"
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers = len(weights)
    layer_indices = np.arange(num_layers)
    
    plt.figure(figsize=(12, 6))
    plt.bar(layer_indices, weights, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Weight Value', fontsize=14)
    plt.title(f'Learned Layer Weights Distribution\n{model_name} on {dataset_name}', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 통계 정보 추가
    stats_text = f'Min: {np.min(weights):.4f}\nMax: {np.max(weights):.4f}\nMean: {np.mean(weights):.4f}\nStd: {np.std(weights):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'weight_distribution_{model_name}_{dataset_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved weight distribution plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    pass
