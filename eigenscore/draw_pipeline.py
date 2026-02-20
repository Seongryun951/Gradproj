import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_paper_pipeline_no_arrows():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # 공통 스타일 (arrow_props는 제거됨)
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    header_style = dict(fontsize=12, fontweight='bold', bbox=dict(facecolor='#E6F3FF', edgecolor='none', pad=3))

    # --- PHASE 1: Data Generation ---
    ax.add_patch(patches.Rectangle((2, 60), 30, 37, fill=True, color='#F9F9F9', alpha=0.3, linestyle='--', linewidth=2))
    ax.text(17, 94, "PHASE 1: Data Generation & Storage", ha='center', **header_style)
    
    ax.text(17, 88, "CoQA / NQ Dataset", ha='center', bbox=box_style)
    
    # LLM Box
    ax.text(17, 78, "LLM Decoder\n(K=10 Sampling)", ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6CC', edgecolor='#D58B00', linewidth=1.5))
    
    # Hidden States
    ax.text(17, 68, "Extract Hidden States\n(All Layers)", ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1D5E7', edgecolor='#9673A6'))
    
    # Layer-wise EigenScores
    ax.text(17, 60, "Layer-wise EigenScores\n[E₀, E₁, ..., Eₗ]", ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#D5E8D4', edgecolor='#82B366', linewidth=1.5))

    # Save step
    ax.text(17, 55, "Save to .pkl\nresultDict", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8CECC', edgecolor='#B85450'))

    # --- PHASE 2: Regression Training ---
    ax.add_patch(patches.Rectangle((35, 25), 30, 72, fill=True, color='#F0FFF0', alpha=0.3, linestyle='--', linewidth=2))
    ax.text(50, 94, "PHASE 2: Regression-based Weight Learning", ha='center', **header_style)

    # Data Source (from P1)
    ax.text(50, 85, "Load resultDict\n(.pkl file)", ha='center', bbox=box_style)
    
    # Ground Truth
    ax.text(50, 75, "Ground Truth Labeling\n(ROUGE-L > 0.5 → Label=1)", ha='center', fontsize=9, bbox=box_style)
    
    # Feature Matrix
    ax.text(50, 65, "Feature Matrix X\n(N × L)\nLabel Vector y\n(N,)", ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#D5E8D4', edgecolor='#82B366', linewidth=1.5))
    
    # Train/Test Split
    ax.text(50, 55, "Train/Test Split\n(80% / 20%)", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1D5E7', edgecolor='#9673A6'))
    
    # Model
    ax.text(50, 45, "Regression Model\n(ElasticNet / Ridge)", ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#DAE8FC', edgecolor='#6C8EBF', linewidth=1.5))
    
    # Learned Weights
    ax.text(50, 33, "Learned Weights\nw = [w₀, w₁, ..., wₗ]", ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF2CC', edgecolor='#D6B656', linewidth=2))
    
    # Visual (Bar Chart small)
    ax.add_patch(patches.Rectangle((42, 25), 16, 6, fill=True, facecolor='white', edgecolor='gray', linewidth=1))
    ax.text(50, 28, "Weight Distribution", fontsize=7, ha='center', fontweight='bold')
    for i in range(1, 14, 2):
        h = 1.5 + (i % 4) * 0.5
        ax.add_patch(patches.Rectangle((43+i, 25.5), 1.0, h, facecolor='#6C8EBF', edgecolor='#4A7C9E'))

    # --- PHASE 3: Evaluation ---
    ax.add_patch(patches.Rectangle((68, 45), 30, 52, fill=True, color='#FFF0F5', alpha=0.5, linestyle='--'))
    ax.text(83, 94, "PHASE 3: Evaluation", ha='center', **header_style)

    # Weighted Sum
    ax.text(83, 85, "Weighted Sum\nS(x) = Σ wl * El", ha='center', bbox=box_style)
    
    # Performance Metrics
    ax.text(83, 72, "Performance Metrics\nAUROC, PCC", ha='center', bbox=box_style)
    
    # Baseline Comparison
    ax.text(83, 58, "VS Baseline\n(Simple Average)", ha='center', bbox=dict(boxstyle='round', facecolor='#F8CECC', edgecolor='#B85450'))

    plt.tight_layout()
    plt.savefig('hallucination_detection_pipeline_no_arrows.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    draw_paper_pipeline_no_arrows()