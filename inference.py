import argparse
from utils.preprocess_data import get_data
from utils.utils import compute_corpus_level_bleu_score, eval_ppl
from models.rnn_nmt import RnnNMT


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained NMT model')
    parser.add_argument('--model_path', type=str, default='checkpoints/att_dot_product_tf1_0_dec_beam/model_rnn_best.pt',
                        help='Path to model file (default: save_dir/model_rnn_best.pt)')
    parser.add_argument('--data_path', type=str, 
                        default='/data/250010009/course/nlpAllms/data/translation_dataset_zh_en',
                        help='Path to data directory')
    parser.add_argument('--decoding_strategy', type=str, default='beam-search',
                        choices=['greedy', 'beam-search'],
                        help='Decoding strategy: greedy or beam-search (default: beam-search)')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for beam-search decoding (default: 5)')
    parser.add_argument('--max_decoding_time_step', type=int, default=50,
                        help='Maximum decoding time steps (default: 50)')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    model = RnnNMT.load(args.model_path)
    
    print(f"Loading data from: {args.data_path}")
    (src_train_sents, tgt_train_sents,
        src_dev_sents, tgt_dev_sents,
        src_test_sents, tgt_test_sents) = get_data(args.data_path)
    
    # Prepare test data for evaluation
    test_data_list = list(zip(src_test_sents, tgt_test_sents))
    
    # Compute perplexity
    print("Computing perplexity on test set..")
    test_ppl = eval_ppl(model, test_data_list)
    print(f"Test Perplexity: {test_ppl:.2f}")
    
    # Compute BLEU scores
    print("\nComputing corpus level BLEU score..")
    print(f"Using decoding strategy: {args.decoding_strategy}")
    if args.decoding_strategy == 'beam-search':
        print(f"Beam size: {args.beam_size}")
    
    test_data = [src_test_sents, tgt_test_sents]
    bleu_scores = compute_corpus_level_bleu_score(
        model, test_data,
        decoding_strategy=args.decoding_strategy,
        beam_size=args.beam_size,
        max_decoding_time_step=args.max_decoding_time_step
    )
    print(f"Corpus BLEU-1: {bleu_scores['bleu_1']*100:.3f}%, BLEU-2: {bleu_scores['bleu_2']*100:.3f}%, BLEU-3: {bleu_scores['bleu_3']*100:.3f}%, BLEU-4: {bleu_scores['bleu_4']*100:.3f}%")


    # Example translation
    print("\n" + "="*80)
    print("Example Translation:")
    print("="*80)
    src_sent = src_test_sents[0]
    gold_sent = tgt_test_sents[0]
    
    if args.decoding_strategy == 'beam-search':
        print(f"Using beam_search (beam_size={args.beam_size}):")
        hyps = model.beam_search(src_sent, beam_size=args.beam_size, max_decoding_time_step=args.max_decoding_time_step)
        if hyps:
            best_hyp = hyps[0]
            hyp_tokens = best_hyp.value
            valid_tokens = [t for t in hyp_tokens if t not in ['<s>', '</s>', '<unk>', '<pad>'] and t.strip()]
            try:
                hyp_sentence = model.sp.decode(valid_tokens)
                hyp_sentence = hyp_sentence.replace('⁇', '').replace('', '').strip()
            except:
                hyp_sentence = " ".join([t.replace('▁', '') if t.startswith('▁') else t for t in valid_tokens])
            print(f"Source: {src_sent}")
            print(f"Reference: {gold_sent}")
            print(f"Generated (beam_search): {valid_tokens}")
            print(f"Generated sentence: {hyp_sentence}")
            print(f"Score: {best_hyp.score:.2f}")
        else:
            print("No hypotheses generated")
    else:
        print("Using greedy_decoding:")
        tgt_token, tgt_sent = model.greedy_decoding(src_sent, max_length=args.max_decoding_time_step)
        print(f"Source: {src_sent}")
        print(f"Reference: {gold_sent}")
        print(f"Generated (greedy): {tgt_token}")
        print(f"Generated sentence: {tgt_sent}")


if __name__ == "__main__":
    main()


