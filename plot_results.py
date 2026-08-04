import os
import re
import sys
import matplotlib.pyplot as plt

def parse_arr(s):
    # Parses numpy printed array strings like "[0.00020192 0.00040192 0.00109947]"
    clean_s = s.replace('[', '').replace(']', '').replace(',', ' ').strip()
    return [float(x) for x in clean_s.split()]

def parse_and_plot(log_path, save_path="training_metrics.png"):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    epochs, losses = [], []
    recalls, ndcgs, precisions, hit_ratios = [], [], [], []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line_str = line.strip()
        if line_str.startswith('|') and re.match(r'\|\s*\d+', line_str):
            parts = [p.strip() for p in line_str.split('|')[1:-1]]
            if len(parts) >= 8:
                try:
                    epoch = int(parts[0])
                    loss = float(parts[3])
                    recall_arr = parse_arr(parts[4])
                    ndcg_arr = parse_arr(parts[5])
                    prec_arr = parse_arr(parts[6])
                    hr_arr = parse_arr(parts[7])
                    
                    epochs.append(epoch)
                    losses.append(loss)
                    recalls.append(recall_arr)
                    ndcgs.append(ndcg_arr)
                    precisions.append(prec_arr)
                    hit_ratios.append(hr_arr)
                except Exception as e:
                    continue

    if not epochs:
        print(f"No valid evaluation metrics found in log file: {log_path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training Loss
    axes[0, 0].plot(epochs, losses, label='Loss', color='crimson', marker='o')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Default Ks in parser: [10, 20, 50]
    ks_labels = [10, 20, 50]

    # 2. Recall
    num_k = len(recalls[0])
    for i in range(num_k):
        k_val = ks_labels[i] if i < len(ks_labels) else (i+1)*10
        r_vals = [r[i] for r in recalls]
        axes[0, 1].plot(epochs, r_vals, label=f'Recall@{k_val}', marker='o')
    axes[0, 1].set_title('Recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Recall Score')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # 3. NDCG
    for i in range(num_k):
        k_val = ks_labels[i] if i < len(ks_labels) else (i+1)*10
        n_vals = [n[i] for n in ndcgs]
        axes[1, 0].plot(epochs, n_vals, label=f'NDCG@{k_val}', marker='s', linestyle='--')
    axes[1, 0].set_title('NDCG')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('NDCG Score')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # 4. Hit Ratio
    for i in range(num_k):
        k_val = ks_labels[i] if i < len(ks_labels) else (i+1)*10
        hr_vals = [h[i] for h in hit_ratios]
        axes[1, 1].plot(epochs, hr_vals, label=f'Hit Ratio@{k_val}', marker='^', linestyle='-.')
    axes[1, 1].set_title('Hit Ratio')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Hit Ratio Score')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plots successfully saved to '{save_path}' across {len(epochs)} evaluation epoch(s).")

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else './results/tripartite_aug_ttv/0mrl_dns16_[4, 8, 16, 32, 64].txt'
    parse_and_plot(log_file)
