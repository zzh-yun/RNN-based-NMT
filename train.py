import sys
import time
import os
import json
from tqdm import tqdm
import torch
import numpy as np
import hydra
from typing import Optional
from omegaconf import DictConfig
from torch import dropout
from models.rnn_nmt import RnnNMT
from dataset.vocab import Vocab
from utils.utils import fix_random_seeds, batch_iter, eval_ppl, compute_corpus_level_bleu_score
from utils.preprocess_data import get_data
import logging

# import debugpy
# #保证host和端口一致，listen可以只设置端口。则为localhost,否则设置成(host,port)
# debugpy.listen(17171)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")


# create a logger
logger = logging.getLogger("Model Recoder")


def train(model, data, learning_rate, lr_decay, clip_grad, batch_size, max_epochs,
          max_num_trial, patience_limit, model_save_path, metrics_file=None,
          decoding_strategy="beam-search", beam_size=5):
    """Run optimization to train the model. Save the best performing model parameters.

    Args:
        model (RnnNMT): RnnNMT object.
        data (Dict): A dataset object.
        learning_rate (float): A scalar giving the learning rate.
        lr_decay (float): A scalar for exponentially decaying the learning rate.
        clip_grad (float): A scalar for gradient clipping.
        batch_size (int): Size of minibatches used to compute loss and gradient during training.
        max_epochs (int): The number of epochs to run for during training.
        max_num_trial (int): The number of trials before termination.
        patience_limit (int): The number of epochs to wait before returning to the best model.
        model_save_path (str): File path to save the model.
        metrics_file (str, optional): File path to save training metrics JSON file.
    """
    model.train()
    
    # Initialize metrics history for tracking
    metrics_history = []

    # Check if 'cuda' is available.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = xm.xla_device() # tpu
    print("Using device: %s" % device)

    # Send the model to device.
    model = model.to(device)

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    
    # Learning rate warmup: use smaller LR for first few epochs
    warmup_epochs = 3
    warmup_factor = 0.1  # Start with 10% of learning rate

    total_samples = len(data['train_data'])
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    # Generate paths for best and last models
    dir_name = os.path.dirname(model_save_path)
    base_name = os.path.basename(model_save_path)
    name, ext = os.path.splitext(base_name)
    best_model_path = os.path.join(dir_name, f"{name}_best{ext}") if dir_name else f"{name}_best{ext}"
    last_model_path = os.path.join(dir_name, f"{name}_last{ext}") if dir_name else f"{name}_last{ext}"
    
    # Try to load optimizer state from last model if exists (for resuming training)
    last_optim_path = last_model_path + ".optim"
    if os.path.exists(last_optim_path):
        print(f"Loading optimizer state from {last_optim_path}...")
        try:
            optimizer.load_state_dict(torch.load(last_optim_path, map_location=device))
            print("  Optimizer state loaded successfully")
        except:
            print("  Failed to load optimizer state, starting with fresh optimizer")
    
    # Begin training.
    print("Begin training..")
    epoch = patience = num_trial = 0
    best_dev_ppl = 0.
    while True:
        tic = time.time()
        epoch += 1
        
        # Learning rate warmup: gradually increase LR for first few epochs
        if epoch <= warmup_epochs:
            warmup_lr = learning_rate * (warmup_factor + (1.0 - warmup_factor) * epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Initialize accumulation variables OUTSIDE the batch loop
        report_loss = report_examples = cum_tgt_words = 0.
        grad_norms = []  # Track gradient norms
        
        for src_sents, tgt_sents in tqdm(
            batch_iter(data["train_data"], batch_size, shuffle=True),
            total=total_batches,
            desc="Tranining",
            unit="batch"):
            
            curr_batch_size = len(src_sents)

            # Compute the forward pass and the loss.
            total_loss = model(src_sents, tgt_sents)
            loss = total_loss / curr_batch_size

            # Zero the gradients, perform backward pass, clip the gradients, and update the gradients.
            optimizer.zero_grad()
            loss.backward()
            
            # Monitor gradient norms before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            grad_norms.append(total_norm)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            # Bookkeeping.
            report_loss += total_loss.item()
            report_examples += curr_batch_size
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)    # omitting leading "<s>"
            cum_tgt_words += tgt_word_num_to_predict

        # At the end of every epoch evaluate the model perplexity on the development set.
        avg_loss = report_loss / report_examples
        train_ppl = np.exp(report_loss / cum_tgt_words)
        dev_ppl = eval_ppl(model, data["dev_data"])
        valid_bleu_scores = compute_corpus_level_bleu_score(
            model, data["dev_data"],
            decoding_strategy=decoding_strategy,
            beam_size=beam_size
        )
        valid_bleu = valid_bleu_scores['bleu_4']  # 使用BLEU-4作为主要指标
        
        toc = time.time()

        # Printout results.
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        max_grad_norm = np.max(grad_norms) if grad_norms else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        train_time = (toc-tic)/60
        # Display BLEU scores as percentage (multiply by 100) in log
        logger.info("Epoch (%d/%d), train time: %.2f min, avg loss: %.1f, avg train ppl: %.1f, dev ppl: %.1f, dev bleu-1: %.2f, bleu-2: %.2f, bleu-3: %.2f, bleu-4: %.2f, lr: %.2e, grad_norm: %.2f (max: %.2f)" % (
            epoch, max_epochs, train_time, avg_loss, train_ppl, dev_ppl, 
            valid_bleu_scores['bleu_1'] * 100, valid_bleu_scores['bleu_2'] * 100, 
            valid_bleu_scores['bleu_3'] * 100, valid_bleu_scores['bleu_4'] * 100, 
            current_lr, avg_grad_norm, max_grad_norm))
        
        # Record metrics for visualization
        metrics = {
            'epoch': epoch,
            'loss': float(avg_loss),
            'train_ppl': float(train_ppl),
            'dev_ppl': float(dev_ppl),
            'bleu_1': float(valid_bleu_scores['bleu_1']),
            'bleu_2': float(valid_bleu_scores['bleu_2']),
            'bleu_3': float(valid_bleu_scores['bleu_3']),
            'bleu_4': float(valid_bleu),
            'lr': float(current_lr),
            'grad_norm': float(avg_grad_norm),
            'max_grad_norm': float(max_grad_norm),
            'train_time': float(train_time)
        }
        metrics_history.append(metrics)
        
        # Save metrics to file (update after each epoch for real-time visualization)
        if metrics_file:
            try:
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_history, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save metrics: {e}")
        
        # Fix: Add generation samples to check model learning progress
        if epoch % 5 == 0 or epoch == 1:  # Print samples every 5 epochs or on first epoch
            print("\n" + "="*80)
            print(f"Generation Samples (Epoch {epoch}):")
            print("="*80)
            model.eval()
            with torch.no_grad():
                # Sample a few examples from dev set
                sample_indices = [0, min(10, len(data["dev_data"])-1), min(20, len(data["dev_data"])-1)]
                for idx in sample_indices:
                    if idx < len(data["dev_data"]):
                        src_sent, tgt_sent = data["dev_data"][idx]
                        # Generate translation
                        hyps = model.beam_search(src_sent, beam_size=2, max_decoding_time_step=50)
                        if hyps:
                            best_hyp = hyps[0]
                            hyp_tokens = best_hyp.value
                            # Fix: Filter invalid tokens before decoding
                            valid_hyp_tokens = [t for t in hyp_tokens if t not in ['<s>', '</s>', '<unk>', '<pad>'] and t.strip()]
                            try:
                                hyp_sentence = model.sp.decode(valid_hyp_tokens)
                                # Fix: Remove replacement characters
                                hyp_sentence = hyp_sentence.replace('⁇', '').replace('', '').strip()
                            except:
                                hyp_sentence = " ".join([t.replace('▁', '') if t.startswith('▁') else t for t in valid_hyp_tokens])
                            
                            # Decode reference
                            ref_tokens = tgt_sent[1:-1] if len(tgt_sent) > 1 and tgt_sent[0] == "<s>" else tgt_sent
                            valid_ref_tokens = [t for t in ref_tokens if t not in ['<s>', '</s>', '<unk>', '<pad>'] and t.strip()]
                            try:
                                ref_sentence = model.sp.decode(valid_ref_tokens)
                                ref_sentence = ref_sentence.replace('⁇', '').replace('', '').strip()
                            except:
                                ref_sentence = " ".join([t.replace('▁', '') if t.startswith('▁') else t for t in valid_ref_tokens])
                            
                            print(f"\nExample {idx+1}:")
                            print(f"  Source: {' '.join(src_sent[:15])}...")
                            print(f"  Reference: {ref_sentence[:100]}...")
                            print(f"  Generated: {hyp_sentence[:100]}...")
                            print(f"  Raw tokens: {hyp_tokens[:20]}...")  # Show raw tokens for debugging
                            print(f"  Score: {best_hyp.score:.2f}")
            model.train()
            print("="*80 + "\n")

        # Save last model at the end of every epoch
        print(f"saving the last model (epoch {epoch}) to [{last_model_path}]..")
        model.save(last_model_path)
        torch.save(optimizer.state_dict(), last_model_path + ".optim")

        # If the model is performing better than it was on the previous epoch, then save
        # the model parameters and the optimizer state as best model.
        # If the model is performing worse than it was on the previous epoch, then
        # increase the patience. if the patience reaches a limit decay the learning rate
        # and increase trial count.
        if epoch%5==0 and (best_dev_ppl == 0 or dev_ppl < best_dev_ppl):
            best_dev_ppl = dev_ppl
            print(f"saving the new best model (epoch {epoch}, dev_ppl: {dev_ppl:.2f}) to [{best_model_path}]..")
            model.save(best_model_path)
            torch.save(optimizer.state_dict(), best_model_path + ".optim")
            patience = 0
        else:
            patience += 1
            print("increasing patience = %d" % patience)
            if patience >= patience_limit:
                # Reset patience.
                patience = 0

                # Increase the trial count.
                num_trial += 1
                print("increasing num trial: %d" % num_trial)

                # Decay the learning rate.
                lr_scheduler.step()

        # If the trial count reaches the maximum number of trials, stop the training.
        if num_trial >= max_num_trial:
            print("Reached maximum number of trials!")
            break

        # If the maximum number of epochs is reached, stop the training.
        if epoch == max_epochs:
            print("Reached maximum number of epochs!")
            break

    # Load the best saved model.
    if os.path.exists(best_model_path):
        print(f"loading the best performing model from [{best_model_path}]..")
        params = torch.load(best_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params["state_dict"])
        model.train()
        model = model.to(device)
    else:
        print(f"Warning: Best model not found at {best_model_path}, using last model instead.")
        if os.path.exists(last_model_path):
            print(f"loading the last model from [{last_model_path}]..")
            params = torch.load(last_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params["state_dict"])
    model.train()
    model = model.to(device)



def generate_experiment_name(cfgs):
    """Generate experiment directory name from configuration parameters.
    
    Args:
        cfgs: Configuration dictionary
        
    Returns:
        str: Experiment directory name
    """
    # If experiment_name is specified, use it
    if cfgs.get('experiment_name') is not None and cfgs.get('experiment_name'):
        return str(cfgs['experiment_name'])
    
    # Otherwise, auto-generate from config parameters
    attention_type = cfgs.get('attention_type', 'dot-product')
    # Replace special characters in attention_type for directory name
    attention_type_clean = attention_type.replace('-', '_').replace(' ', '_')
    
    teacher_forcing_ratio = cfgs.get('teacher_forcing_ratio', 0.7)
    # Format teacher forcing ratio (1.0 -> tf1.0, 0.7 -> tf0.7)
    tf_str = f"tf{teacher_forcing_ratio:.1f}".replace('.', '_')
    
    decoding_strategy = cfgs.get('decoding_strategy', 'beam-search')
    # Format decoding strategy (beam-search -> beam, greedy -> greedy)
    if decoding_strategy == 'beam-search':
        dec_str = 'beam'
    else:
        dec_str = 'greedy'
    
    experiment_name = f"att_{attention_type_clean}_{tf_str}_dec_{dec_str}"
    return experiment_name


@hydra.main(version_base='1.3', config_path='./configs', config_name='train.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    print('='*30,'Training Parameters:','='*30,'\n')
    logger.info(cfgs)
    if cfgs.get('seed') is not None: fix_random_seeds(cfgs.get('seed'))

    # Generate experiment name and update output paths
    experiment_name = generate_experiment_name(cfgs)
    print(f"Experiment name: {experiment_name}")

    print('='*30,'Loading data...','='*30,'\n')
    (src_train_sents, tgt_train_sents,   # The sents means the sentence
     src_valid_sents, tgt_valid_sents,
     src_test_sents, tgt_test_sents
    ) = get_data(cfgs.get('data_root'))

    print('='*30,'Building a vocabulary of source and target language...','='*30,'\n')

    vocab = Vocab.build(src_train_sents, tgt_train_sents, cfgs['vocab_size'], cfgs['freq_cutoff'])
    vocab.save(cfgs['vocav_save_dir'])

    # if we already have a vocabulary cache, we can just load it
    # vocab = vocab.load(cfgs['vocav_save_dir'])
    print('='*30,'Bulding a model...','='*30,'\n')
    
    # Check if loading from existing model
    resume_from_model = cfgs.get('resume_from_model', None)
    if resume_from_model and os.path.exists(resume_from_model):
        print(f"Loading model from {resume_from_model}...")
        model = RnnNMT.load(resume_from_model)
        print("  Model loaded successfully")
    else:
        model = RnnNMT(word_embed_size=cfgs['word_embed_dim'],
                       hidden_size=cfgs['hidden_size'], vocab=vocab, dropout_rate=cfgs['dropout_rate'],
                       teacher_forcing_ratio=cfgs['teacher_forcing_ratio'],
                       frequency_penalty=cfgs.get('frequency_penalty', 0.5),
                       attention_type=cfgs.get('attention_type', 'dot-product'))
        if resume_from_model:
            print(f"Warning: Model file {resume_from_model} not found. Creating new model.")
    
    print('='*30,'Training','='*30,'\n')
    train_data = list(zip(src_train_sents, tgt_train_sents))
    dev_data = list(zip(src_valid_sents, tgt_valid_sents))
    dataset = {"train_data" : train_data, "dev_data" : dev_data}

    # Get output directory from Hydra (automatically created as outputs/{timestamp}/)
    # Hydra sets the working directory to outputs/{timestamp}/, so we can use current directory
    current_dir = os.getcwd()
    
    # Update model save directory to include experiment name
    original_model_save_dir = cfgs['model_save_dir']
    model_save_dir_base = os.path.dirname(original_model_save_dir) if os.path.dirname(original_model_save_dir) else '.'
    model_save_name = os.path.basename(original_model_save_dir)
    # Create experiment-specific model save directory
    experiment_model_dir = os.path.join(model_save_dir_base, experiment_name)
    os.makedirs(experiment_model_dir, exist_ok=True)
    experiment_model_save_path = os.path.join(experiment_model_dir, model_save_name)
    
    metrics_file = os.path.join(current_dir, 'training_metrics.json')
    print(f"Training metrics will be saved to: {metrics_file}")
    print(f"Model will be saved to: {experiment_model_save_path}")

    # Get decoding strategy and beam size from config
    decoding_strategy = cfgs.get('decoding_strategy', 'beam-search')
    beam_size = cfgs.get('beam_size', 5)
    
    tic = time.time()
    # train(xxx)
    train(model, dataset, learning_rate=cfgs['learning_rate'], lr_decay=cfgs['lr_decay'],
        clip_grad=cfgs['clip_grad'], batch_size=cfgs['batch_size'], max_epochs=cfgs['max_epochs'],
        max_num_trial=cfgs['max_num_trial'], patience_limit=cfgs['patience_limit'],
        model_save_path=experiment_model_save_path, metrics_file=metrics_file,
        decoding_strategy=decoding_strategy, beam_size=beam_size)
    toc= time.time()

    print(f"Training took {((toc - tic) / 60):.3f} minutes")


    # Compute and print BLEU score.
    print("Computing corpuse level BLEU score..")
    test_data = [src_test_sents, tgt_test_sents]
    
    print(f"Using decoding strategy: {decoding_strategy}")
    if decoding_strategy == 'beam-search':
        print(f"Beam size: {beam_size}")
    
    tic = time.time()
    bleu_scores = compute_corpus_level_bleu_score(
        model=model, 
        data=test_data,
        decoding_strategy=decoding_strategy,
        beam_size=beam_size
    )
    toc = time.time()
    print(f"Corpus BLEU-1: {bleu_scores['bleu_1']*100:.3f}%, BLEU-2: {bleu_scores['bleu_2']*100:.3f}%, BLEU-3: {bleu_scores['bleu_3']*100:.3f}%, BLEU-4: {bleu_scores['bleu_4']*100:.3f}%. Computed in {toc-tic:.3f} seconds.")

    


if __name__ == "__main__":
    main()