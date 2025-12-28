import math
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu

# ------------------------------- data generation ---------------------------------------#
def batch_iter(data, batch_size, shuffle=False):
    """Yield batches of source and target sentences reverse sorted by length
    (longest to shortest).

    Args:
        data (List[Tuple[List[str], List[str]]]): A list of tuples containing source and
            target sentence.
        batch_size (int): Batch size.
        shuffle (boolean, optional): Whether to randomly shuffle the dataset.
            Defaults to False.

    Yields:
        src_sents (List[List[str]]): A list of source sentences.
        tgt_sents (List[List[str]]): A list of target sentences.
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        yield src_sents, tgt_sents


# ------------------------------- data preparation utilities -----------------------------#
def pad_sents_char(sents, char_pad_token):
    """Pad a list of sentences according to the longest sentence in the batch and the
    longest word in all sentences. The paddings are at the end of each word and at the end
    of each sentence.

    Args:
        sents (List[List[List[int]]]): List of sentences, where each sentence is
            represented as a list of words, and each word is represented as a list of characters.
            result of "words2charindices()" from "vocab.py".
        char_pad_token (int): Index of the character-padding token.

    Returns:
        sents_padded (List[List[List[int]]]): List of sentences where sentences/words
            shorter than the max length sentence/word are padded out with the appropriate
            pad token, such that each sentence in the batch now has same number of words
            and each word has an equal number of characters.
            output shape: (batch_size, max_sentence_length, max_word_length)
    """
    sents_padded = []
    max_word_length = max(len(w) for s in sents for w in s)
    max_sent_len = max(len(s) for s in sents)

    for s in sents:
        # Pad shorer words. Extend shorter sentences.
        s_pad = [[c for c in w] + [char_pad_token for _ in range(max_word_length-len(w))] for w in s]
        s_pad.extend([[char_pad_token] * max_word_length] * max(0, max_sent_len - len(s_pad)))
        sents_padded.append(s_pad)
    
    return sents_padded

def pad_sents(sents, pad_token):
    """Pad a list of sentences according to the longest sentence in the batch. The
    paddings are at the end of each sentence.

    Args:
        sents (List[List[str]]): List of sentences, where each sentence is represented as
            a list of words.
        pad_token (str): Padding token.

    Returns:
        sents_padded (List[List[str]]): List of sentences where sentences shorter than the
            max length sentence are padded out with the pad_token, such that each sentence
            in the batch now has equal length.
    """
    sents_padded = []
    max_length = max(len(s) for s in sents)

    for s_ in sents:
        s = s_.copy()
        s.extend([pad_token] * max(0, max_length - len(s)))
        sents_padded.append(s)
    
    return sents_padded


# ------------------------------- metrics evaluation utilities --------------------------# 
def eval_ppl(model, data, batch_size=64):
    """Evaluate the model perplexity on the given set.

    Args:
        model (NMT): NMT object.
        data (List[Tuple[List[str], List[str]]]): List of tuples containing src and tgt sentence.
        batch_size (int, optional): Size of the batch to evaluate perplexity on.
            Defaults to 64.

    Returns:
        ppl (float): Perplexity on the given sentences.
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.           # cumulative loss
    cum_tgt_words = 0.      # cumulative number of target words

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(data, batch_size):
            loss = model(src_sents, tgt_sents)

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading "<s>"
            cum_tgt_words += tgt_word_num_to_predict

    ppl = np.exp(cum_loss / cum_tgt_words)
    if was_training: model.train()
    return ppl
    
# ∏ i=14precisioni：把 1 元、2 元、3 元、4 元语法的精度相乘，体现 “多粒度匹配”

def compute_corpus_level_bleu_score(model, data, decoding_strategy="beam-search", beam_size=2, max_decoding_time_step=50): # Changed from 10 to 70 for proper sentence generation
    """Evaluate the model corpus-level BLEU score on the given set.

    Args:
        model (NMT): Trained NMT model.
        data (Tuple(src_sent, tgt_sent)): Tuple containing source sentences and target sentences.
        decoding_strategy (str, optional): Decoding strategy, either "greedy" or "beam-search". Defaults to "beam-search".
        beam_size (int, optional): Number of hypotheses to hold for a translation at every
            step. Only used when decoding_strategy is "beam-search". Defaults to 2.
        max_decoding_time_step (int, optional): Maximum sentence length that decoding
            can produce. Defaults to 50.

    Returns:
        bleu_score (dict): Dictionary containing BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.
    """
    was_training = model.training
    model.eval()
    
    source_sentences = [item[0] for item in data]      # Input sentences for translation.
    references = [item[1] for item in data]           # Gold-standard reference target sentences.
    
    # Fix: Data validation - check if references are in correct format (should be English SPM tokens)
    # If references don't contain SPM tokens (starting with ▁), they might be Chinese
    if references and len(references) > 0:
        sample_ref = references[0]
        # Check if reference contains SPM tokens (should start with ▁ for English)
        has_spm_tokens = any(token.startswith('▁') for token in sample_ref if isinstance(token, str))
        # Check if reference contains Chinese characters (common Chinese characters)
        has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in token) for token in sample_ref if isinstance(token, str))
        
        if has_chinese and not has_spm_tokens:
            print("="*80)
            print("WARNING: References appear to be Chinese instead of English!")
            print(f"Sample source: {source_sentences[0][:10] if source_sentences else 'None'}...")
            print(f"Sample reference: {sample_ref[:10]}...")
            print("This suggests data order might be wrong. Expected: (Chinese_source, English_target)")
            print("="*80)
    
    # Fix: Remove <s> and </s> tokens from references if present
    if references and len(references[0]) > 0 and references[0][0] == "<s>":
        references = [ref[1:-1] if len(ref) > 1 and ref[-1] == "</s>" else ref[1:] for ref in references]

    # Run decoding to construct hypotheses for a list of src-language sentences.
    hypotheses = []
    with torch.no_grad():
        for src_sent in source_sentences:
            if decoding_strategy == "greedy":
                # Use greedy decoding
                hyp_tokens, hyp_sentence = model.greedy_decoding(src_sent, max_length=max_decoding_time_step)
                # Create a Hypothesis-like object for consistency
                class SimpleHypothesis:
                    def __init__(self, value, score=0.0):
                        self.value = value
                        self.score = score
                hypotheses.append([SimpleHypothesis(value=hyp_tokens, score=0.0)])
            else:
                # Use beam search (default)
                example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                                 max_decoding_time_step=max_decoding_time_step)
                hypotheses.append(example_hyps)
    top_hypotheses = [hyps[0] if hyps else None for hyps in hypotheses]

    
    # Fix: Process hypotheses - decode SPM tokens to words
    processed_hypotheses = []
    for hyp in top_hypotheses:
        if hyp is None:
            processed_hypotheses.append([])
            continue
        # 1. 提取hyp的value（带▁的SPM token列表，如['▁the', '▁is']）
        hyp_tokens = hyp.value
        # Fix: Filter out invalid tokens before decoding
        # Remove special tokens and invalid tokens that SPM can't decode
        valid_tokens = []
        for token in hyp_tokens:
            if token in ['<s>', '</s>', '<unk>', '<pad>']:
                continue
            # Check if token is valid (not empty, not just special characters)
            if token and token.strip():
                valid_tokens.append(token)
        
        # 2. 用SPM的decode还原为自然句子（如"the is"）
        if len(valid_tokens) > 0:
            try:
                hyp_sentence = model.sp.decode(valid_tokens)
                # Fix: Clean up the decoded sentence - remove replacement characters
                # SPM may produce "⁇" (U+2047) or other replacement characters for invalid tokens
                hyp_sentence = hyp_sentence.replace('⁇', '').replace('', '').strip()
                # 3. 拆分为单词列表（和参考文本粒度一致）
                hyp_word_list = hyp_sentence.split() if hyp_sentence else []
                # Filter out empty strings
                hyp_word_list = [w for w in hyp_word_list if w.strip()]
            except Exception as e:
                # Fallback: if decode fails, use tokens directly (remove SPM prefix if present)
                hyp_word_list = [token.replace('▁', '') if token.startswith('▁') else token 
                                for token in valid_tokens if token.replace('▁', '').strip()]
        else:
            hyp_word_list = []
        processed_hypotheses.append(hyp_word_list)

    # Fix: Process references - decode SPM tokens to words
    # References should be English SPM tokens (from target language)
    processed_references = []
    for ref_idx, ref in enumerate(references):
        if not ref:
            processed_references.append([])
            continue
        
        # Check if ref contains Chinese characters (data might be wrong)
        has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in str(token)) for token in ref if token)
        if has_chinese:
            # This is a data error - reference should be English, not Chinese
            print(f"WARNING: Reference {ref_idx} contains Chinese characters! This is incorrect for EN target.")
            print(f"  Reference tokens: {ref[:10]}...")
            # Skip this reference or use empty list
            processed_references.append([])
            continue
        
        # Check if ref contains SPM tokens (starting with ▁) - should be the case for English
        has_spm_tokens = any(token.startswith('▁') for token in ref if isinstance(token, str))
        
        if has_spm_tokens:
            # SPM tokens, need to decode
            try:
                ref_sentence = model.sp.decode(ref)
                ref_word_list = ref_sentence.strip().split() if ref_sentence.strip() else []
            except Exception as e:
                # Fallback: remove SPM prefix manually
                ref_word_list = [token.replace('▁', '') if token.startswith('▁') else token 
                                for token in ref if token not in ['<s>', '</s>', '<unk>', '<pad>']]
        else:
            # No SPM tokens - might be already word-level tokens (shouldn't happen for English)
            # Try to decode anyway in case they're valid English words
            try:
                # If tokens look like words, use them directly
                ref_word_list = [token for token in ref if token not in ['<s>', '</s>', '<unk>', '<pad>'] and token.strip()]
            except:
                ref_word_list = []
        
        processed_references.append(ref_word_list)

    # ========== 确保参考文本格式符合corpus_bleu要求 ==========
    # corpus_bleu要求：references是「列表的列表」（每个样本的参考列表，即使只有1个参考）
    formatted_references = [[ref] for ref in processed_references]

    # Fix: Add debug output to diagnose BLEU=0 issue
    # Print first few examples to see what's being generated
    print("\n" + "="*80)
    print("BLEU Debug Info - First 5 examples:")
    print("="*80)
    for i in range(min(5, len(processed_hypotheses))):
        hyp_tokens_raw = top_hypotheses[i].value if top_hypotheses[i] else []
        hyp_words = processed_hypotheses[i]
        ref_words = processed_references[i]
        source_tokens = source_sentences[i][:15] if i < len(source_sentences) else []
        print(f"\nExample {i+1}:")
        print(f"  Source: {' '.join(source_tokens)}...")
        print(f"  Raw hyp tokens: {hyp_tokens_raw[:20]}...")  # First 20 tokens
        print(f"  Hyp words: {' '.join(hyp_words[:20])}...")  # First 20 words
        print(f"  Ref words: {' '.join(ref_words[:20])}...")  # First 20 words
        print(f"  Hyp length: {len(hyp_words)}, Ref length: {len(ref_words)}")
        if len(hyp_words) == 0:
            print("  WARNING: Generated hypothesis is empty!")
        if len(ref_words) == 0:
            print("  WARNING: Reference is empty!")
        # Check for repetition
        if len(hyp_words) > 5:
            unique_ratio = len(set(hyp_words)) / len(hyp_words)
            if unique_ratio < 0.5:
                print(f"  WARNING: Low diversity (unique ratio: {unique_ratio:.2f}) - possible repetition!")
    print("="*80 + "\n")

    # 计算Corpus-level BLEU分数
    # 分别计算BLEU-1, BLEU-2, BLEU-3, BLEU-4
    try:
        # BLEU-1: 只考虑1-gram
        bleu_1 = corpus_bleu(formatted_references, processed_hypotheses, weights=(1.0, 0, 0, 0))
        
        # BLEU-2: 考虑1-gram和2-gram
        bleu_2 = corpus_bleu(formatted_references, processed_hypotheses, weights=(0.5, 0.5, 0, 0))
        
        # BLEU-3: 考虑1-gram, 2-gram和3-gram
        bleu_3 = corpus_bleu(formatted_references, processed_hypotheses, weights=(1/3, 1/3, 1/3, 0))
        
        # BLEU-4: 考虑1-gram到4-gram（默认）
        bleu_4 = corpus_bleu(formatted_references, processed_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        
        # 输出详细的BLEU分数（以百分比形式显示）
        print("\n" + "="*80)
        print("BLEU Scores Breakdown:")
        print("="*80)
        print(f"  BLEU-1: {bleu_1*100:.4f}% (unigram precision)")
        print(f"  BLEU-2: {bleu_2*100:.4f}% (bigram precision)")
        print(f"  BLEU-3: {bleu_3*100:.4f}% (trigram precision)")
        print(f"  BLEU-4: {bleu_4*100:.4f}% (4-gram precision, standard BLEU)")
        print("="*80 + "\n")
        
        # 返回字典包含所有BLEU分数（0-1范围，保持原始值）
        bleu_scores = {
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
        
    except Exception as e:
        print(f"ERROR in corpus_bleu calculation: {e}")
        print(f"Sample formatted_ref: {formatted_references[0] if formatted_references else 'None'}")
        print(f"Sample processed_hyp: {processed_hypotheses[0] if processed_hypotheses else 'None'}")
        bleu_scores = {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0
        }
        print("\n" + "="*80)
        print("BLEU Scores Breakdown:")
        print("="*80)
        print(f"  BLEU-1: 0.0000% (calculation failed)")
        print(f"  BLEU-2: 0.0000% (calculation failed)")
        print(f"  BLEU-3: 0.0000% (calculation failed)")
        print(f"  BLEU-4: 0.0000% (calculation failed)")
        print("="*80 + "\n")
    
    if was_training: model.train()
    
    # 返回包含所有BLEU分数的字典（0-1范围）
    # 注意：所有分数都是0-1范围，需要显示时乘以100
    return bleu_scores

# -------------------------------- helper functions -------------------------------------#
def fix_random_seeds(seed):
    """Manually set the seed for random number generation.
    Also set CuDNN flags for reproducible results using deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
