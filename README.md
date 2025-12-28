# RNN-based Neural Machine Translation (NMT)

A PyTorch implementation of RNN-based Neural Machine Translation system for Chinese-to-English translation, featuring LSTM encoder-decoder architecture with attention mechanisms.

⭐⭐⭐Because the github cannot upload big files, the checkpoints of the model is upload to huggingface. If you want to use our model, please download the checkpoints from huggingface:  https://huggingface.co/DarcyCheng/RNN-based-NMT

## Introduction

This repository implements a RNN-based Neural Machine Translation system with the following key components:

**Model**: Implement a model using LSTM, with both the encoder and decoder consisting of unidirectional layers.

**Attention mechanism**: Implement the attention mechanism and investigate the impact of different alignment functions—such as dot-product, multiplicative, and additive—on model performance.

**Training policy**: Compare the effectiveness of Teacher Forcing and Free Running strategies.

**Decoding policy**: Compare the effectiveness of greedy and beam-search decoding strategies.

### Key Features

- **Encoder**: Unidirectional LSTM encoder for source language (Chinese)
- **Decoder**: Unidirectional LSTM decoder with attention mechanism for target language (English)
- **Attention Types**: 
  - Dot-product attention
  - Multiplicative attention
  - Additive attention (Bahdanau-style)
- **Tokenization**:
  - Chinese: Jieba word segmentation
  - English: SentencePiece subword tokenization
- **Training Strategies**:
  - Teacher Forcing (configurable ratio)
  - Free Running
- **Decoding Strategies**:
  - Greedy decoding
  - Beam search decoding (configurable beam size)

## Data Preparation

The compressed package contains four JSONL files, corresponding respectively to the small training set, large training set, validation set, and test set, with sizes of 100k, 10k, 500, and 200 samples. Each line in a JSONL file contains one parallel sentence pair. The final model performance will be evaluated based on results on the test set.

### Data Format

Each line in the JSONL files follows this format:
```json
{"chinese": "中文句子", "english": "English sentence"}
```

### Data Directory Structure

```
translation_dataset_zh_en/
├── train_small.jsonl      # 100k samples
├── train_large.jsonl      # 10k samples  
├── dev.jsonl              # 500 samples
└── test.jsonl             # 200 samples
```

### Preprocessing

The data preprocessing pipeline includes:
1. Chinese text segmentation using Jieba
2. English text tokenization using SentencePiece
3. Vocabulary construction with frequency cutoff
4. Sentence padding and batching

## Environment

### Requirements

- **Python**: Python 3.9.25
- **PyTorch**: torch 2.0.1+cu118 (or compatible version)
- **CUDA**: CUDA 11.8 (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RNN_NMT
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

3. Download NLTK data (required for BLEU score calculation):
```python
import nltk
nltk.download('punkt')
```

### Dependencies

Key dependencies include:
- `torch>=1.12.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `hydra-core>=1.3.0` - Configuration management
- `omegaconf>=2.2.0` - Configuration objects
- `sentencepiece>=0.1.96` - English subword tokenization
- `jieba>=0.42.1` - Chinese word segmentation
- `nltk>=3.7` - BLEU score evaluation
- `tqdm>=4.62.0` - Progress bars

## Training and Evaluation

### Training

Train the model using the default configuration:

```bash
python train.py
```

The training script uses Hydra for configuration management. You can override configuration parameters via command line:

```bash
python train.py attention_type=additive teacher_forcing_ratio=0.7 decoding_strategy=beam-search beam_size=5
```

### Configuration

Main training parameters can be configured in `configs/train.yaml`:

- `attention_type`: "dot-product", "multiplicative", or "additive"
- `teacher_forcing_ratio`: Ratio for teacher forcing (0.0-1.0)
- `decoding_strategy`: "greedy" or "beam-search"
- `beam_size`: Beam size for beam search (default: 5)
- `learning_rate`: Initial learning rate (default: 5e-5)
- `batch_size`: Batch size (default: 128)
- `max_epochs`: Maximum training epochs (default: 50)

### Evaluation

Evaluate a trained model on the test set:

```bash
python eval.py
```

Or with custom parameters:

```bash
python eval.py --model_path <path_to_model> --data_path <path_to_data> --decoding_strategy beam-search --beam_size 5
```

Alternatively, you can use `inference.py` directly (same functionality):

```bash
python inference.py --model_path <path_to_model> --data_path <path_to_data> --decoding_strategy beam-search --beam_size 5
```

The evaluation script will output:
- Perplexity (PPL) on test set
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
- Detailed translation examples

### Model Checkpoints

During training, the model saves:
- **Best model**: `save_dir/model_rnn_best.pt` (best validation perplexity)
- **Last model**: `save_dir/model_rnn_last.pt` (most recent checkpoint)
- **Optimizer state**: Saved alongside model files (`.optim` extension)

### Resuming Training

To resume training from a checkpoint:

```yaml
# In configs/train.yaml
resume_from_model: "save_dir/model_rnn_last.pt"
```

## Project Structure

```
RNN_NMT/
├── configs/
│   └── train.yaml          # Training configuration
├── dataset/
│   └── vocab.py            # Vocabulary management
├── models/
│   ├── rnn_nmt.py          # Main NMT model
│   ├── model_embeddings.py # Embedding layers
│   └── char_decoder.py     # Character-level decoder
├── utils/
│   ├── utils.py            # Utility functions (BLEU, batching, etc.)
│   └── preprocess_data.py  # Data preprocessing
├── train.py                # Training script
├── inference.py            # Evaluation script
├── eval.py                 # Evaluation script (alias for inference.py)
├── requirement.txt         # Python dependencies
└── README.md              # This file
```

## Experimental Results

The model performance is evaluated using:
- **Perplexity (PPL)**: Lower is better
- **BLEU Score**: Higher is better (BLEU-4 as primary metric)

Training metrics are automatically saved to `training_metrics.json` for visualization and analysis.

## Acknowledgement

感谢以下几个仓库：

1. **Jieba** (Chinese word segmentation tool): [https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba)

2. **SentencePiece** (English and multilingual subword tokenization tool): [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

3. **RNN Machine Translation**: [https://github.com/pi-tau/machine-translation](https://github.com/pi-tau/machine-translation)

## License

[Add your license information here]

## Contact

[Add your contact information here]

