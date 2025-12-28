"""
Training Visualization from Log File
Parse training log file and create visualization
"""

import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_log_file(log_file):
    """Parse training log file and extract metrics
    
    Args:
        log_file: Path to training log file
    
    Returns:
        metrics: List of metric dictionaries
    """
    metrics = []
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return metrics
    
    # Pattern to match: Epoch (X/Y), train time: X min, avg loss: X, avg train ppl: X, dev ppl: X, dev bleu-1: X, bleu-2: X, bleu-3: X, bleu-4: X, lr: X, grad_norm: X (max: X)
    pattern = r'Epoch \((\d+)/(\d+)\), train time: ([\d.]+) min, avg loss: ([\d.]+), avg train ppl: ([\d.]+), dev ppl: ([\d.]+), dev bleu-1: ([\d.]+), bleu-2: ([\d.]+), bleu-3: ([\d.]+), bleu-4: ([\d.]+), lr: ([\d.e+-]+), grad_norm: ([\d.]+) \(max: ([\d.]+)\)'
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                max_epochs = int(match.group(2))
                train_time = float(match.group(3))
                avg_loss = float(match.group(4))
                train_ppl = float(match.group(5))
                dev_ppl = float(match.group(6))
                bleu_1 = float(match.group(7))
                bleu_2 = float(match.group(8))
                bleu_3 = float(match.group(9))
                bleu_4 = float(match.group(10))
                lr = float(match.group(11))
                grad_norm = float(match.group(12))
                max_grad_norm = float(match.group(13))
                
                metrics.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'train_ppl': train_ppl,
                    'dev_ppl': dev_ppl,
                    'bleu_1': bleu_1,
                    'bleu_2': bleu_2,
                    'bleu_3': bleu_3,
                    'bleu_4': bleu_4,
                    'lr': lr,
                    'grad_norm': grad_norm,
                    'max_grad_norm': max_grad_norm,
                    'train_time': train_time
                })
    
    return metrics


def create_visualization(metrics, output_dir):
    """Create visualization from metrics data
    
    Args:
        metrics: List of metric dictionaries
        output_dir: Directory to save the figures
    """
    if not metrics:
        print("No metrics to visualize")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    epochs = [m['epoch'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    train_ppls = [m['train_ppl'] for m in metrics]
    dev_ppls = [m['dev_ppl'] for m in metrics]
    bleu_1s = [m['bleu_1'] for m in metrics]
    bleu_2s = [m['bleu_2'] for m in metrics]
    bleu_3s = [m['bleu_3'] for m in metrics]
    bleu_4s = [m['bleu_4'] for m in metrics]
    lrs = [m['lr'] for m in metrics]
    
    # 1. BLEU Scores (all in one figure)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(epochs, bleu_1s, 'b-', linewidth=2, label='BLEU-1', marker='o', markersize=5, alpha=0.8)
    ax1.plot(epochs, bleu_2s, 'g-', linewidth=2, label='BLEU-2', marker='s', markersize=5, alpha=0.8)
    ax1.plot(epochs, bleu_3s, 'r-', linewidth=2, label='BLEU-3', marker='^', markersize=5, alpha=0.8)
    ax1.plot(epochs, bleu_4s, 'm-', linewidth=2, label='BLEU-4', marker='d', markersize=5, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('BLEU Score (%)', fontsize=14, fontweight='bold')
    ax1.set_title('BLEU Scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12, loc='best')
    
    # Add statistics
    if bleu_4s:
        max_bleu_4 = max(bleu_4s)
        max_epoch_bleu_4 = epochs[bleu_4s.index(max_bleu_4)]
        ax1.text(0.02, 0.98, f'Max BLEU-4: {max_bleu_4:.2f}% @ Epoch {max_epoch_bleu_4}', 
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    bleu_path = os.path.join(output_dir, 'bleu_scores.png')
    plt.savefig(bleu_path, dpi=300, bbox_inches='tight')
    print(f"BLEU scores visualization saved to: {bleu_path}")
    plt.close()
    
    # 2. Loss
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(epochs, losses, 'b-', linewidth=2, label='Loss', marker='o', markersize=5, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Training Loss', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12)
    
    # Add statistics
    if losses:
        min_loss = min(losses)
        min_epoch = epochs[losses.index(min_loss)]
        ax2.axvline(x=min_epoch, color='r', linestyle='--', alpha=0.5, linewidth=2)
        ax2.text(0.02, 0.98, f'Min Loss: {min_loss:.2f} @ Epoch {min_epoch}', 
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Loss visualization saved to: {loss_path}")
    plt.close()
    
    # 3. Perplexity
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.plot(epochs, train_ppls, 'g-', linewidth=2, label='Train PPL', marker='o', markersize=5, alpha=0.8)
    ax3.plot(epochs, dev_ppls, 'r-', linewidth=2, label='Dev PPL', marker='s', markersize=5, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax3.set_title('Train vs Dev Perplexity', fontsize=16, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=12)
    
    # Add statistics
    if dev_ppls:
        min_dev_ppl = min(dev_ppls)
        min_epoch = epochs[dev_ppls.index(min_dev_ppl)]
        ax3.axvline(x=min_epoch, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax3.text(0.02, 0.98, f'Best Dev PPL: {min_dev_ppl:.2f} @ Epoch {min_epoch}', 
                transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    ppl_path = os.path.join(output_dir, 'perplexity.png')
    plt.savefig(ppl_path, dpi=300, bbox_inches='tight')
    print(f"Perplexity visualization saved to: {ppl_path}")
    plt.close()
    
    # 4. Learning Rate
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.plot(epochs, lrs, 'c-', linewidth=2, label='Learning Rate', marker='o', markersize=5, alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax4.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold', pad=15)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, linestyle='--', which='both')
    ax4.legend(fontsize=12)
    
    # Add statistics
    if lrs:
        max_lr = max(lrs)
        min_lr = min(lrs)
        max_epoch = epochs[lrs.index(max_lr)]
        min_epoch = epochs[lrs.index(min_lr)]
        ax4.text(0.02, 0.98, f'Max LR: {max_lr:.2e} @ Epoch {max_epoch}\nMin LR: {min_lr:.2e} @ Epoch {min_epoch}', 
                transform=ax4.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    lr_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(lr_path, dpi=300, bbox_inches='tight')
    print(f"Learning rate visualization saved to: {lr_path}")
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"Total epochs visualized: {len(metrics)}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training progress from log file')
    parser.add_argument('--log_file', type=str, default='train.log',
                        help='Path to training log file (default: train.log)')
    parser.add_argument('--output_dir', type=str, default='save_dir',
                        help='Output directory to save figures (default: save_dir)')
    
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    metrics = parse_log_file(args.log_file)
    
    if not metrics:
        print("No metrics found in log file")
        return
    
    print(f"Found {len(metrics)} epochs in log file")
    
    # Create visualization
    print("Creating visualizations...")
    create_visualization(metrics, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
