"""
Embedding PCA Visualization Script
Visualize train/valid/test dataset embeddings in 2D space after PCA dimensionality reduction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.decomposition import PCA
from models.rnn_nmt import RnnNMT
from utils.preprocess_data import get_data


def get_word_embeddings(model, sentences, vocab_type='src', max_words=None):
    """Get word embeddings from model embedding layer
    
    Args:
        model: Trained RnnNMT model
        sentences: List of sentences (list of lists of tokens)
        vocab_type: 'src' for source language (Chinese) or 'tgt' for target language (English)
        max_words: Maximum number of unique words to process (None for all)
    
    Returns:
        embeddings: numpy array of shape (n_words, word_embed_size)
        words: list of words corresponding to embeddings
    """
    # Collect all unique words from sentences
    word_set = set()
    for sentence in sentences:
        for word in sentence:
            if word and word.strip():  # Filter out empty strings
                word_set.add(word)
    
    words = list(word_set)
    
    if max_words is not None and len(words) > max_words:
        # Sample randomly
        indices = np.random.choice(len(words), max_words, replace=False)
        words = [words[i] for i in indices]
        print(f"Sampling {max_words} words from {len(word_set)} total unique words")
    
    model.eval()
    device = model.device
    
    # Select vocabulary and embedding layer based on vocab_type
    if vocab_type == 'src':
        vocab = model.vocab.src
        embedding_layer = model.model_embeddings_source.embed
        lang_name = "Chinese (source)"
    else:  # vocab_type == 'tgt'
        vocab = model.vocab.tgt
        embedding_layer = model.model_embeddings_target.embed
        lang_name = "English (target)"
    
    # Get word IDs
    word_ids = []
    valid_words = []
    for word in words:
        if word in vocab.word2id:
            word_id = vocab.word2id[word]
            word_ids.append(word_id)
            valid_words.append(word)
    
    if not word_ids:
        print(f"Warning: No valid words found in {lang_name} vocabulary")
        return np.array([]), []
    
    # Get embeddings from embedding layer
    with torch.no_grad():
        word_ids_tensor = torch.tensor(word_ids, device=device)
        # embedding_layer: (vocab_size, word_embed_size)
        # word_ids_tensor: (n_words,)
        word_embeddings = embedding_layer(word_ids_tensor)
        # word_embeddings: (n_words, word_embed_size)
        embeddings = word_embeddings.cpu().numpy()
    
    print(f"  Extracted {len(valid_words)} {lang_name} words")
    return embeddings, valid_words


def visualize_pca_embeddings(train_emb, valid_emb, test_emb, output_path, 
                             train_words=None, valid_words=None, test_words=None):
    """Visualize word embeddings in 2D PCA space
    
    Args:
        train_emb: Training set word embeddings (n_words, word_embed_size)
        valid_emb: Validation set word embeddings (n_words, word_embed_size)
        test_emb: Test set word embeddings (n_words, word_embed_size)
        output_path: Path to save the visualization
        train_words: List of words in training set (optional)
        valid_words: List of words in validation set (optional)
        test_words: List of words in test set (optional)
    """
    # Check if any dataset is empty
    if len(train_emb) == 0 or len(valid_emb) == 0 or len(test_emb) == 0:
        print("Error: One or more datasets have no embeddings!")
        print(f"  Train: {len(train_emb)} words")
        print(f"  Valid: {len(valid_emb)} words")
        print(f"  Test: {len(test_emb)} words")
        return
    
    # Combine all embeddings for PCA fitting
    all_embeddings = np.vstack([train_emb, valid_emb, test_emb])
    
    # Fit PCA on all embeddings
    print("Fitting PCA...")
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    
    # Transform each dataset
    print("Transforming embeddings...")
    train_2d = pca.transform(train_emb)
    valid_2d = pca.transform(valid_emb)
    test_2d = pca.transform(test_emb)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    print(f"PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}, Total={total_variance:.2%}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each dataset with different colors and markers
    scatter1 = ax.scatter(train_2d[:, 0], train_2d[:, 1], 
                        c='#3498db', alpha=0.6, s=20, 
                        label=f'Train ({len(train_2d)} samples)', edgecolors='none')
    
    scatter2 = ax.scatter(valid_2d[:, 0], valid_2d[:, 1], 
                        c='#e74c3c', alpha=0.6, s=20, 
                        label=f'Valid ({len(valid_2d)} samples)', edgecolors='none')
    
    scatter3 = ax.scatter(test_2d[:, 0], test_2d[:, 1], 
                        c='#2ecc71', alpha=0.6, s=20, 
                        label=f'Test ({len(test_2d)} samples)', edgecolors='none')
    
    # Add labels and title
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=12, fontweight='bold')
    # Determine language name for title
    lang_label = "Chinese" if train_words and any(any('\u4e00' <= c <= '\u9fff' for c in w) for w in train_words[:10]) else "English"
    ax.set_title(f'{lang_label} Word Embeddings in 2D PCA Space\n(Train/Valid/Test Datasets)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with statistics
    stats_text = f'Total Explained Variance: {total_variance:.2%}\n'
    stats_text += f'Train: {len(train_2d)} words\n'
    stats_text += f'Valid: {len(valid_2d)} words\n'
    stats_text += f'Test: {len(test_2d)} words'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize sentence embeddings in 2D PCA space')
    parser.add_argument('--model_path', type=str, default='save_dir/model_rnn.pt',
                        help='Model file path (default: save_dir/model_rnn.pt)')
    parser.add_argument('--data_path', type=str, 
                        default='/data/250010009/course/nlpAllms/data/translation_dataset_zh_en',
                        help='Data root path')
    parser.add_argument('--output_path', type=str, default='save_dir/embedding_pca_visualization.png',
                        help='Output image path (default: save_dir/embedding_pca_visualization.png)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of unique words per dataset (None for all, default: None)')
    parser.add_argument('--lang', type=str, default='src', choices=['src', 'tgt'],
                        help='Language to visualize: "src" for Chinese (source) or "tgt" for English (target), default: "src"')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Load model
    print(f"Loading model: {args.model_path}")
    model = RnnNMT.load(args.model_path)
    model.eval()
    print("Model loaded successfully!")
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    (src_train_sents, tgt_train_sents,
     src_valid_sents, tgt_valid_sents,
     src_test_sents, tgt_test_sents) = get_data(args.data_path)
    
    print(f"Data loaded:")
    print(f"  Train: {len(src_train_sents)} sentences")
    print(f"  Valid: {len(src_valid_sents)} sentences")
    print(f"  Test: {len(src_test_sents)} sentences")
    
    # Select sentences based on language type
    if args.lang == 'src':
        train_sents = src_train_sents
        valid_sents = src_valid_sents
        test_sents = src_test_sents
        lang_name = "Chinese (source)"
    else:  # args.lang == 'tgt'
        train_sents = tgt_train_sents
        valid_sents = tgt_valid_sents
        test_sents = tgt_test_sents
        lang_name = "English (target)"
    
    # Get word embeddings for each dataset
    print(f"\nExtracting {lang_name} word embeddings from train dataset...")
    train_emb, train_words = get_word_embeddings(model, train_sents, 
                                                 vocab_type=args.lang,
                                                 max_words=args.max_samples)
    
    print(f"Extracting {lang_name} word embeddings from valid dataset...")
    valid_emb, valid_words = get_word_embeddings(model, valid_sents, 
                                                 vocab_type=args.lang,
                                                 max_words=args.max_samples)
    
    print(f"Extracting {lang_name} word embeddings from test dataset...")
    test_emb, test_words = get_word_embeddings(model, test_sents, 
                                              vocab_type=args.lang,
                                              max_words=args.max_samples)
    
    print(f"\nEmbedding shapes:")
    print(f"  Train: {train_emb.shape} ({len(train_words)} words)")
    print(f"  Valid: {valid_emb.shape} ({len(valid_words)} words)")
    print(f"  Test: {test_emb.shape} ({len(test_words)} words)")
    
    # Create visualization
    print("\nCreating visualization...")
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    visualize_pca_embeddings(train_emb, valid_emb, test_emb, args.output_path,
                             train_words, valid_words, test_words)
    print("Done!")


if __name__ == "__main__":
    main()

