"""
Word Frequency Distribution Visualization Script
Visualize original word frequency distribution and frequency penalty distribution
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import os
import argparse
import re
from models.rnn_nmt import RnnNMT

# Function to find and set Chinese font
def setup_chinese_font():
    """Find and set a font that supports Chinese characters"""
    # First, try to use font file path directly (most reliable method)
    common_font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/Windows/Fonts/simhei.ttf',
    ]
    
    for font_path in common_font_paths:
        if os.path.exists(font_path):
            try:
                # Use FontProperties with file path directly
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                # Set font using the file path method
                plt.rcParams['font.sans-serif'] = [font_name]
                # Try to add font to font manager if method exists
                try:
                    if hasattr(fm.fontManager, 'addfont'):
                        fm.fontManager.addfont(font_path)
                except:
                    pass
                print(f"Using Chinese font from file: {font_path} (font name: {font_name})")
                plt.rcParams['axes.unicode_minus'] = False
                return
            except Exception as e:
                print(f"Failed to load font from {font_path}: {e}")
                continue
    
    # If font file method didn't work, try to find by name
    # Clear matplotlib font cache to force refresh
    try:
        import matplotlib
        cache_dir = matplotlib.get_cachedir()
        import glob
        cache_files = glob.glob(os.path.join(cache_dir, 'fontlist*.json'))
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
            except:
                pass
    except:
        pass
    
    # Try to find fonts that support Chinese
    chinese_font_names = [
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',  # Put these first since we installed them
        'SimHei', 'Microsoft YaHei',
        'Noto Sans CJK SC', 'Source Han Sans CN', 'STHeiti', 'STSong',
        'Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'Droid Sans Fallback'
    ]
    
    # Get all available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Try to find a Chinese font
    chinese_font = None
    for font_name in chinese_font_names:
        if font_name in available_fonts:
            chinese_font = font_name
            break
    
    # If no Chinese font found, try to find any font that might support CJK
    if chinese_font is None:
        # Check fonts that might support CJK characters
        for font in fm.fontManager.ttflist:
            font_name = font.name
            # Some fonts have CJK in their name or are known to support CJK
            if any(keyword in font_name.lower() for keyword in ['wqy', 'wenquanyi', 'cjk', 'noto', 'source', 'droid']):
                chinese_font = font_name
                break
    
    # Set the font
    if chinese_font:
        # Clear any existing font settings and set the Chinese font as primary
        plt.rcParams['font.sans-serif'] = [chinese_font]
        print(f"Using Chinese font: {chinese_font}")
    else:
        # Last resort: try to use a font that might work
        # Use DejaVu Sans as fallback, but warn the user
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        print("="*80)
        print("WARNING: No Chinese font found on the system!")
        print("Chinese characters may not display correctly in the visualization.")
        print("\nTo fix this, you can:")
        print("1. Install a Chinese font (e.g., wqy-microhei):")
        print("   sudo apt-get install fonts-wqy-microhei  # Ubuntu/Debian")
        print("   sudo yum install wqy-microhei-fonts      # CentOS/RHEL")
        print("2. Or download a font file and place it in a system font directory")
        print("="*80)
    
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def is_chinese_char(char):
    """Check if a character is Chinese"""
    return '\u4e00' <= char <= '\u9fff'


def is_chinese_word(word):
    """Check if a word contains Chinese characters"""
    return any(is_chinese_char(c) for c in word)


def is_english_word(word):
    """Check if a word is English (contains only ASCII letters)"""
    return bool(re.match(r'^[a-zA-Z]+$', word))


def get_special_token_ids(vocab):
    """Get special token IDs"""
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>']
    special_token_ids = []
    for token in special_tokens:
        if token in vocab.word2id:
            special_token_ids.append(vocab.word2id[token])
    return set(special_token_ids)


def extract_frequency_data(model):
    """Extract word frequency and penalty data from model"""
    vocab_tgt = model.vocab.tgt
    vocab_src = model.vocab.src
    frequency_penalty = model.frequency_penalty
    penalty_scale = 10.0  # Consistent with model's penalty_scale
    
    special_token_ids_tgt = get_special_token_ids(vocab_tgt)
    special_token_ids_src = get_special_token_ids(vocab_src)
    
    # Extract target language (English) word frequencies
    word_frequencies = []
    penalties = []
    word_ids = []
    word_freq_dict_tgt = {}  # word -> frequency for top words (target)
    
    for word_id in vocab_tgt.word_freq:
        if word_id not in special_token_ids_tgt:
            # Original word frequency
            freq = vocab_tgt.word_freq.get(word_id, 0)
            word_frequencies.append(freq)
            
            # Calculate penalty value
            normalized_freq = vocab_tgt.normalized_freq.get(word_id, 0.0)
            penalty = frequency_penalty * normalized_freq * penalty_scale
            penalties.append(penalty)
            
            word_ids.append(word_id)
            
            # Store word and frequency for top words analysis
            word = vocab_tgt.id2word.get(word_id, '')
            if word:
                word_freq_dict_tgt[word] = freq
    
    # Extract source language (Chinese) word frequencies
    word_freq_dict_src = {}  # word -> frequency for top words (source)
    
    for word_id in vocab_src.word_freq:
        if word_id not in special_token_ids_src:
            freq = vocab_src.word_freq.get(word_id, 0)
            word = vocab_src.id2word.get(word_id, '')
            if word:
                word_freq_dict_src[word] = freq
    
    return word_frequencies, penalties, word_ids, frequency_penalty, word_freq_dict_tgt, word_freq_dict_src


def get_top_words(word_freq_dict, top_n=20):
    """Get top N words by frequency, separated by English and Chinese"""
    # Separate English and Chinese words
    english_words = [(word, freq) for word, freq in word_freq_dict.items() if is_english_word(word)]
    chinese_words = [(word, freq) for word, freq in word_freq_dict.items() if is_chinese_word(word)]
    
    # Sort by frequency
    english_words.sort(key=lambda x: x[1], reverse=True)
    chinese_words.sort(key=lambda x: x[1], reverse=True)
    
    return english_words[:top_n], chinese_words[:top_n]


def print_statistics(word_frequencies, penalties, frequency_penalty, word_freq_dict_tgt, word_freq_dict_src):
    """Print statistics and top words"""
    print("\n" + "="*80)
    print("Word Frequency Statistics")
    print("="*80)
    print(f"\nOriginal Word Frequency Distribution (Target Language):")
    print(f"  Total vocabulary size: {len(word_frequencies)}")
    print(f"  Min: {np.min(word_frequencies):.2f}")
    print(f"  Max: {np.max(word_frequencies):.2f}")
    print(f"  Mean: {np.mean(word_frequencies):.2f}")
    print(f"  Median: {np.median(word_frequencies):.2f}")
    print(f"  Std: {np.std(word_frequencies):.2f}")
    
    print(f"\nFrequency Penalty Distribution (frequency_penalty={frequency_penalty}):")
    print(f"  Min: {np.min(penalties):.4f}")
    print(f"  Max: {np.max(penalties):.4f}")
    print(f"  Mean: {np.mean(penalties):.4f}")
    print(f"  Median: {np.median(penalties):.4f}")
    print(f"  Std: {np.std(penalties):.4f}")
    
    # Print top English words from target vocabulary
    top_english, _ = get_top_words(word_freq_dict_tgt, top_n=20)
    
    print(f"\nTop 20 English Words by Frequency (Target Language):")
    for i, (word, freq) in enumerate(top_english, 1):
        print(f"  {i:2d}. {word:20s} : {freq:8d}")
    
    # Print top Chinese words from source vocabulary
    top_chinese_src = [(word, freq) for word, freq in word_freq_dict_src.items() if is_chinese_word(word)]
    top_chinese_src.sort(key=lambda x: x[1], reverse=True)
    top_chinese_src = top_chinese_src[:20]
    
    if top_chinese_src:
        print(f"\nTop 20 Chinese Words by Frequency (Source Language):")
        for i, (word, freq) in enumerate(top_chinese_src, 1):
            print(f"  {i:2d}. {word:20s} : {freq:8d}")
    else:
        print(f"\nNo Chinese words found in source vocabulary")
    
    print("="*80 + "\n")


def create_visualization(word_frequencies, penalties, frequency_penalty, output_path, word_freq_dict_tgt, word_freq_dict_src):
    """Create bar chart visualization for top words"""
    top_english, _ = get_top_words(word_freq_dict_tgt, top_n=20)
    
    # Get top Chinese words from source vocabulary
    top_chinese_src = [(word, freq) for word, freq in word_freq_dict_src.items() if is_chinese_word(word)]
    top_chinese_src.sort(key=lambda x: x[1], reverse=True)
    top_chinese_src = top_chinese_src[:20]
    
    # Determine number of subplots based on available data
    num_plots = 0
    if top_english:
        num_plots += 1
    if top_chinese_src:
        num_plots += 1
    
    if num_plots == 0:
        print("No words to visualize")
        return
    
    # Create subplots
    if num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 7))
        axes = [axes]
    
    plot_idx = 0
    
    # Plot English words
    if top_english:
        ax = axes[plot_idx]
        words = [w[0] for w in top_english]
        freqs = [w[1] for w in top_english]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(words)), freqs, color='#3498db', alpha=0.8, edgecolor='darkblue', linewidth=1.2)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel('Word Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Words', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 English Words by Frequency', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Invert y-axis to show highest frequency at top
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            width = bar.get_width()
            ax.text(width + max(freqs) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{freq:,}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plot_idx += 1
    
    # Plot Chinese words
    if top_chinese_src:
        ax = axes[plot_idx]
        words = [w[0] for w in top_chinese_src]
        freqs = [w[1] for w in top_chinese_src]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(words)), freqs, color='#e74c3c', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)  # Slightly larger font for Chinese characters
        ax.set_xlabel('Word Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Words', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Chinese Words by Frequency', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Invert y-axis to show highest frequency at top
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            width = bar.get_width()
            ax.text(width + max(freqs) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{freq:,}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plot_idx += 1
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show plot (if in interactive environment)
    # plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize word frequency distribution and frequency penalty distribution')
    parser.add_argument('--model_path', type=str, default='save_dir/model_rnn.pt',
                        help='Model file path (default: save_dir/model_rnn.pt)')
    parser.add_argument('--output_path', type=str, default='save_dir/frequency_distribution.png',
                        help='Output image path (default: save_dir/frequency_distribution.png)')
    
    args = parser.parse_args()
    
    # Setup Chinese font at runtime (not at module import)
    print("Setting up Chinese font...")
    setup_chinese_font()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    print(f"Loading model: {args.model_path}")
    model = RnnNMT.load(args.model_path)
    print("Model loaded successfully!")
    
    # Extract data
    print("Extracting word frequency data...")
    word_frequencies, penalties, word_ids, frequency_penalty, word_freq_dict_tgt, word_freq_dict_src = extract_frequency_data(model)
    
    # Print statistics
    print_statistics(word_frequencies, penalties, frequency_penalty, word_freq_dict_tgt, word_freq_dict_src)
    
    # Create visualization
    print("Generating visualization...")
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    create_visualization(word_frequencies, penalties, frequency_penalty, args.output_path, word_freq_dict_tgt, word_freq_dict_src)
    print("Done!")


if __name__ == "__main__":
    main()

