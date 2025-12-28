import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import namedtuple
from .model_embeddings import ModelEmbeddings
import sentencepiece as spm

Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class RnnNMT(nn.Module):
    def __init__(self, word_embed_size, hidden_size, vocab, teacher_forcing_ratio=0.5,
                 dropout_rate=0.2, frequency_penalty=0.5, attention_type='dot-product'):
        """Init NMT Model.

        Args:
            word_embed_size (int): Embedding size for the words.
            hidden_size (int): Hidden Size, the size of hidden states (dimensionality).
            vocab (Vocab): Vocabulary object containing src and tgt languages
                See vocab.py for documentation.
            dropout_rate (float, optional): Dropout probability, for attention.
                Defaults to 0.2.
            attention_type (str, optional): Type of attention mechanism. Options:
                'dot-product', 'multiplicative', 'additive'. Defaults to 'dot-product'.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.teacher_forcing_ratio = teacher_forcing_ratio  # TODO：ablation study
        self.frequency_penalty = frequency_penalty  # Frequency penalty for high-frequency words
        self.attention_type = attention_type  # Attention mechanism type

        # Initialization of the model architecture.
        # Model Embeddings.
        self.model_embeddings_source = ModelEmbeddings(word_embed_size, vocab.src)
        self.model_embeddings_target = ModelEmbeddings(word_embed_size, vocab.tgt)

        # Sequence-to-Sequence with attention architecture.
        # Fix: Unidirectional LSTM should output hidden_size, not hidden_size*2
        self.encoder = nn.LSTM(word_embed_size, hidden_size, bias=True, bidirectional=False)
        self.decoder = nn.LSTMCell(word_embed_size + hidden_size, hidden_size, bias=True)
        # Fix: Update projection layers to match encoder output dimension
        self.h_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Attention mechanism layers - different for each attention type
        if attention_type == 'dot-product':
            # Dot-product attention: project encoder hidden states
            self.att_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == 'multiplicative':
            # Multiplicative attention: project encoder hidden states
            self.att_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == 'additive':
            # Additive attention: need separate projections for encoder and decoder
            self.att_projection_enc = nn.Linear(hidden_size, hidden_size, bias=False)  # W1
            self.att_projection_dec = nn.Linear(hidden_size, hidden_size, bias=False)  # W2
            self.att_v = nn.Linear(hidden_size, 1, bias=False)  # v^T
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}. Must be 'dot-product', 'multiplicative', or 'additive'")
        
        self.combined_output_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)  # dec_hidden (h) + a_t (h) = 2h
        self.dropout = nn.Dropout(dropout_rate)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        ### spm tokenizer
        sp_model_path='save_dir/en_spm_model.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)  # 传入你的SPM模型路径（如en_spm_model.model）
        
        # Fix: Improved parameter initialization
        self._init_parameters()
        
        # Pre-compute frequency penalty tensor to avoid repeated computation in beam_search
        self._precompute_frequency_penalty()
    
    def _init_parameters(self):
        """Initialize model parameters with appropriate methods for different layer types."""
        for name, param in self.named_parameters():
            if 'embed' in name:
                # Embedding layers: normal initialization with larger std for better learning
                torch.nn.init.normal_(param, mean=0.0, std=0.1)  # Increased from 0.01 to 0.1
            elif 'weight' in name:
                if len(param.shape) >= 2:
                    # LSTM/LSTMCell weights: PyTorch stores weights as (4*hidden_size, input_size) for input
                    # and (4*hidden_size, hidden_size) for hidden-to-hidden
                    if 'encoder' in name or 'decoder' in name:
                        # For LSTM encoder/decoder, use orthogonal for hidden-to-hidden, Xavier for input-to-hidden
                        if 'weight_hh' in name:
                            # Hidden-to-hidden weights: use orthogonal initialization
                            # Shape is (4*hidden_size, hidden_size), need to split into 4 parts
                            if param.shape[0] == 4 * param.shape[1]:
                                torch.nn.init.orthogonal_(param)
                            else:
                                torch.nn.init.xavier_uniform_(param, gain=1.0)
                        elif 'weight_ih' in name:
                            # Input-to-hidden weights: use Xavier
                            torch.nn.init.xavier_uniform_(param, gain=1.0)
                        else:
                            # Other weights: Xavier
                            torch.nn.init.xavier_uniform_(param, gain=1.0)
                    else:
                        # Other weight matrices: Xavier/Glorot initialization
                        torch.nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    # Bias or 1D parameters: uniform initialization
                    torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                # Bias terms: zero or small uniform
                # For LSTM, set forget gate bias to 1.0 to help with gradient flow
                if ('encoder' in name or 'decoder' in name) and 'bias' in name:
                    # LSTM bias has shape (4*hidden_size,), set forget gate (2nd quarter) to 1.0
                    if len(param.shape) == 1 and param.shape[0] % 4 == 0:
                        hidden_size = param.shape[0] // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate bias = 1.0
                    else:
                        torch.nn.init.constant_(param, 0.0)
                else:
                    torch.nn.init.constant_(param, 0.0)
            else:
                # Fallback: uniform initialization
                torch.nn.init.uniform_(param, -0.1, 0.1)
    
    def _precompute_frequency_penalty(self):
        """Pre-compute frequency penalty tensor to avoid repeated computation in beam_search.
        
        This method computes the penalty tensor once during initialization, which is much
        faster than computing it for every decoding step.
        """
        if self.frequency_penalty > 0 and hasattr(self.vocab.tgt, 'normalized_freq'):
            vocab_size = len(self.vocab.tgt.word2id)
            penalty_scale = 10.0  # Scale factor to convert normalized frequency to penalty
            penalty_tensor = torch.zeros(vocab_size)
            
            # Compute penalty for each word in vocabulary
            for word_id in range(vocab_size):
                if word_id in self.vocab.tgt.normalized_freq:
                    normalized_freq = self.vocab.tgt.normalized_freq[word_id]
                    # Apply penalty: higher frequency -> larger penalty
                    penalty_tensor[word_id] = self.frequency_penalty * normalized_freq * penalty_scale
            
            # Register as buffer so it moves with the model to GPU/CPU
            self.register_buffer('frequency_penalty_tensor', penalty_tensor)
        else:
            # No penalty if frequency_penalty is 0 or normalized_freq doesn't exist
            self.register_buffer('frequency_penalty_tensor', None)

    def forward(self, source, target):
        """Take a mini-batch of source and target sentences and compute the log-likelihood
        of target sentences under the language model learned by the NMT system.

        Run a forward pass on the network:
            1. Run the input `source_padded` through the encoder.
            2. Generate sentence masks for `source_padded`.
            3. Apply the decoder to compute the decoder outputs.
            4. Compute log-probability distribution over the target vocabulary from the
                decoder outputs.
            5. Compute the word-level loss.

        Args:
            source (List[List[str]]): List of source sentence tokens.
            target (List[List[str]]): List of target sentence tokens. The target sentences
                must be wrapped by `<s>` and `</s>`.

        Returns:
            scores (Tensor): Tensor of shape (b,) representing the log-likelihood of
                generating the gold-standard target sentence for each example in the input
                batch. Here b = batch size.
        """
        # Filter out empty source sentences (and corresponding targets)
        non_empty_indices = [i for i, s in enumerate(source) if len(s) > 0]
        if not non_empty_indices:
            # If no valid sentences, return zero loss to avoid errors
            return torch.tensor(0.0, device=self.device)
        source = [source[i] for i in non_empty_indices]
        target = [target[i] for i in non_empty_indices]
        # Compute sentence lengths.
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors.
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)

        # Compute the scores.
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded, self.teacher_forcing_ratio)
        word_scores = self.target_vocab_projection(combined_outputs)

        # Compute the word-level loss
        loss = F.cross_entropy(word_scores.view(-1, len(self.vocab.tgt)),
            target_padded[1:].view(-1), ignore_index=self.vocab.tgt.word_pad, reduction="sum")

        return loss

    def encode(self, source_padded, source_lengths):
        """Apply the encoder to source sentences to obtain encoder hidden states.
        Additionally, take the final states of the encoder and project them to obtain
        initial states for decoder.

        Run the source sentences through the encoder:
            1. Construct a tensor `X` of source sentences with shape (src_len, b, e) using
                the source model embeddings.
            2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
                - before we can apply the encoder, we need to apply the
                  `pack_padded_sequence` function to `X`
                - after we apply the encoder, we need to apply the `pad_packed_sequence`
                  function to `enc_hiddens`
                - the shape of the tensor returned by the encoder is (src_len, b, h*2) and
                  we want to return a tensor of shape (b, src_len, h*2) as `enc_hiddens`
            3. Compute the initial hidden state of the decoder by applying `h_projection`
                layer to `last_hidden`. Compute the initial cell state of the decoder by
                applying `c_projection` layer to `last_cell`.

        Args:
            source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b),
                where b = batch_size, src_len = maximum source sentence length. Note that
                these have already been sorted in order of longest to shortest sentence.
            source_lengths (List[int]): List of actual lengths for each of the source
                sentences in the batch.

        Returns:
            enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h), where
                b = batch size, src_len = maximum source sentence length, h = hidden size.
            dec_init_state (Tuple(Tensor, Tensor)): Tuple of tensors representing the
                initial hidden state and the initial cell state of the decoder.
        """
        X = self.model_embeddings_source(source_padded)
        packed_X = pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(packed_X)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True)
        # Fix: Unidirectional LSTM outputs (1, batch, hidden_size), so we just squeeze
        # last_hidden and last_cell are already (1, batch, hidden_size) for unidirectional LSTM

        dec_init_hidden = self.h_projection(last_hidden).squeeze(0)
        dec_init_cell = self.c_projection(last_cell).squeeze(0)

        dec_init_state = (dec_init_hidden, dec_init_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded, teacher_forcing_ratio=1):
        """Compute the decoder output vectors for a batch.
        Given the encoder hidden states, the decoder initial state and a batch of target
        sentences, copute the decoder output vectors.
        At each step the input to the decoder is the next item from the target sequence.
        This approach is known as `teacher forcing`.

        Decode the target sentence:
            1. Apply the attention projection layer to `enc_hiddens` to obtain
                `enc_hiddens_proj`, which should be of shape (b, src_len, h),
            2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using
                the target model embeddings.
            3. Use the torch.split function to iterate over the time dimension of Y.
                Within the loop, this will give us Y_t of shape (1, b, e)
                - Squeeze Y_t into a tensor of dimension (b, e). 
                - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
                - Use the step function to compute the the Decoder's next (cell, state)
                  values as well as the new combined output o_t.
                - Append o_t to combined_outputs.
                - Update o_prev to the new o_t.
            4. Use torch.stack to convert combined_outputs from a list of length tgt_len
                of tensors shape (b, h), to a single tensor of shape (tgt_len, b, h).

        Args:
            enc_hiddens (Tensor): Hidden states (b, src_len, h), where
                b = batch size, src_len = maximum source sentence length, h = hidden size.
            enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                b = batch size, src_len = maximum source sentence length.
            dec_init_state (Tuple(Tensor, Tensor)): Initial state and cell for decoder.
            target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b),
                where tgt_len = maximum target sentence length, b = batch size.

        Returns:
            combined_outputs (Tensor): combined output tensor (tgt_len, b, h), where
                tgt_len = maximum target sentence length, b=batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]
        tgt_len = target_padded.shape[0]  # 获取目标序列长度 (tgt_len, b, ...)
        # Initialize the decoder state (hidden and cell).
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero.
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step.
        combined_outputs = []
        
        # Fix: enc_hiddens is now (b, src_len, h), so projection is (h -> h) for attention
        # Project encoder hidden states for attention (different for each attention type)
        if self.attention_type == 'dot-product' or self.attention_type == 'multiplicative':
            enc_hiddens_proj = self.att_projection(enc_hiddens)
        elif self.attention_type == 'additive':
            enc_hiddens_proj = self.att_projection_enc(enc_hiddens)
        else:
            enc_hiddens_proj = self.att_projection(enc_hiddens)  # fallback

        Y = self.model_embeddings_target(target_padded)
        
        # Fix: Get the first input token (usually <s>) - always use ground truth for first token
        Y_t = torch.squeeze(Y[0:1], dim=0)  # (b, e)
        
        for t in range(tgt_len):
            # Concatenate current input with previous output
            Ybar_t = torch.cat((Y_t, o_prev), dim=-1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            
            # Fix: Teacher forcing - decide per timestep, not per sequence
            # For training: randomly decide whether to use ground truth or prediction
            # For evaluation: always use prediction (teacher_forcing_ratio should be 0 or model.eval())
            if self.training and t + 1 < tgt_len:
                # Decide per timestep whether to use teacher forcing
                use_tf = (teacher_forcing_ratio == 1.0 or 
                         torch.rand(1, device=self.device).item() < teacher_forcing_ratio)
                if use_tf:
                    # Use ground truth for next step
                    Y_t = torch.squeeze(Y[t+1:t+2], dim=0)
                else:
                    # Use model prediction for next step
                    word_scores = self.target_vocab_projection(o_t)  # (b, vocab_size)
                    _, predicted_ids = torch.max(word_scores, dim=1)  # (b,)
                    # Embed the predicted tokens
                    Y_t = self.model_embeddings_target(predicted_ids.unsqueeze(0))  # (1, b, e)
                    Y_t = torch.squeeze(Y_t, dim=0)  # (b, e)
            else:
                # Evaluation mode or last timestep: use prediction
                if t + 1 < tgt_len:
                    word_scores = self.target_vocab_projection(o_t)  # (b, vocab_size)
                    _, predicted_ids = torch.max(word_scores, dim=1)  # (b,)
                    Y_t = self.model_embeddings_target(predicted_ids.unsqueeze(0))  # (1, b, e)
                    Y_t = torch.squeeze(Y_t, dim=0)  # (b, e)
            
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs)

        return combined_outputs

    def step(self, Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        """Compute one forward step of the LSTM decoder, including the attention computation.

        Run a single timestep of the Decoder:
            1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
            2. Split dec_state into its two parts (dec_hidden, dec_cell)
            3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
                Note: b = batch_size, src_len = maximum source length, h = hidden size.
            4. Use enc_masks to set the attention score to "-inf" for the padding.
            5. Apply softmax to e_t to yield alpha_t - the attention distribution. 
            6. Compute the attention output vector, a_t, of shape (b, h). The attention
                output vector is the weighted sum of enc_hiddens (b, src_len, h).
            7. Concatenate dec_hidden with a_t to compute tensor U_t (b, 2h).
            8. Apply the combined output projection layer to U_t to compute tensor V_t.
            9. Compute tensor o_t by first applying tanh function and then the dropout layer.

        Args:
            Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h).
                The input for the decoder, where b = batch size, e = embedding size, h = hidden size.
            dec_state (Tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h),
                where b = batch size, h = hidden size. First tensor is decoder's prev
                hidden state, second tensor is decoder's prev cell.
            enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h),
                where b = batch size, src_len = maximum source length, h = hidden size.
            enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected for attention.
                Tensor is with shape (b, src_len, h), where b = batch size,
                src_len = maximum source length, h = hidden size.
            enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                where b = batch size, src_len is maximum source length.

        Returns:
            dec_state (Tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h),
                where b = batch size, h = hidden size. First tensor is decoder's new
                hidden state, second tensor is decoder's new cell state.
            o_t (Tensor): Combined output Tensor at timestep t, shape (b, h), where
                b = batch size, h = hidden size.
            e_t (Tensor): Attention scores distribution Tensor at timestep t, shape (b, src_len).
        """
        dec_state = self.decoder(Ybar_t, dec_state)   # Ybar_t (batch size, embedding size, hidden size)
        (dec_hidden, dec_cell) = dec_state

        # Compute attention scores based on attention type
        if self.attention_type == 'dot-product':
            # Dot-product attention: e_t[i] = enc_hiddens_proj[i]^T * dec_hidden
            # enc_hiddens_proj: (b, src_len, h), dec_hidden: (b, h)
            e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(dim=-1)).squeeze(dim=-1)  # (b, src_len)
        
        elif self.attention_type == 'multiplicative':
            # Multiplicative attention: e_t[i] = (W * enc_hiddens[i])^T * dec_hidden
            # enc_hiddens_proj already contains W * enc_hiddens
            # Same as dot-product but with learned projection
            e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(dim=-1)).squeeze(dim=-1)  # (b, src_len)
        
        elif self.attention_type == 'additive':
            # Additive attention: e_t[i] = v^T * tanh(W1 * enc_hiddens[i] + W2 * dec_hidden)
            # enc_hiddens_proj contains W1 * enc_hiddens: (b, src_len, h)
            # Project decoder hidden state: W2 * dec_hidden: (b, h)
            dec_hidden_proj = self.att_projection_dec(dec_hidden)  # (b, h)
            # Expand dec_hidden_proj to match enc_hiddens_proj: (b, 1, h)
            dec_hidden_proj_expanded = dec_hidden_proj.unsqueeze(1)  # (b, 1, h)
            # Add: (b, src_len, h) + (b, 1, h) -> (b, src_len, h)
            combined = torch.tanh(enc_hiddens_proj + dec_hidden_proj_expanded)  # (b, src_len, h)
            # Apply v^T: (b, src_len, h) @ (h, 1) -> (b, src_len, 1) -> (b, src_len)
            e_t = self.att_v(combined).squeeze(dim=-1)  # (b, src_len)
        
        else:
            # Fallback to dot-product
            e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float("inf"))

        alpha_t = F.softmax(e_t, dim=-1)
        a_t = torch.bmm(alpha_t.unsqueeze(dim=1), enc_hiddens).squeeze(dim=1)
        U_t = torch.cat((dec_hidden, a_t), dim=-1)
        V_t = self.combined_output_projection(U_t)
        o_t = self.dropout(torch.tanh(V_t))
        # o_t = self.dropout(F.tanh(V_t)) # deprecated

        return dec_state, o_t, e_t

    def generate_sent_masks(self, enc_hiddens, source_lengths):
        """Generate sentence masks for encoder hidden states.

        Args:
            enc_hiddens (Tensor): Encodings of shape (b, src_len, h), where
                b = batch size, src_len = max source length, h = hidden size. 
            source_lengths (List[int]): List of actual lengths for each of the sentences
                in the batch.

        Returns:
            enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len), where
                b = batch size, src_len is maximum source length.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def greedy_decoding(self, src_sent, max_length=50):
        """Given a single source sentence, decode the sentence, yielding translation in
        the target language.

        Args:
            src_sent (List[str]): A single source sentence (list of words).
            max_length (int, optional): Maximum number of time steps to unroll the
                decoding RNN. Defaults to 50.

        Returns:
            hypothesis (List[str]): The target sentence represented as a list of words.
        """
        x = self.vocab.src.to_input_tensor([src_sent], self.device)
        enc_hiddens, dec_init_state = self.encode(x, [len(src_sent)])
        
        # Fix: Handle different attention types
        if self.attention_type == 'dot-product' or self.attention_type == 'multiplicative':
            enc_hiddens_proj = self.att_projection(enc_hiddens)
        elif self.attention_type == 'additive':
            enc_hiddens_proj = self.att_projection_enc(enc_hiddens)
        else:
            enc_hiddens_proj = self.att_projection(enc_hiddens)  # fallback

        dec_state = dec_init_state
        o_prev = torch.zeros(1, self.hidden_size, device=self.device)

        hypothesis = ["<s>"]
        min_length = 3  # Minimum length before allowing </s>
        max_reasonable_length = 30  # Maximum reasonable length before forcing EOS
        
        while len(hypothesis) < max_length:
            y_t = self.vocab.tgt.to_input_tensor([[hypothesis[-1]]], device=self.device)
            y_t_embed = self.model_embeddings_target(y_t)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)
            Ybar_t = torch.cat((y_t_embed, o_prev), dim=-1)

            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj,
                                          enc_masks=None)
            log_p_t = F.log_softmax(self.target_vocab_projection(o_t), dim=-1)
            
            # Fix: Apply frequency penalty if available
            if self.frequency_penalty_tensor is not None:
                log_p_t = log_p_t - self.frequency_penalty_tensor.unsqueeze(0).to(log_p_t.device)
            
            # Fix: Penalize </s> if sequence is too short, but allow it after reasonable length
            actual_length = len(hypothesis) - 1  # Excluding <s>
            if actual_length < min_length:
                eos_idx = self.vocab.tgt['</s>']
                log_p_t[:, eos_idx] = log_p_t[:, eos_idx] - 50.0
            elif actual_length >= max_reasonable_length:  # Encourage EOS if sequence is already long
                eos_idx = self.vocab.tgt['</s>']
                log_p_t[:, eos_idx] = log_p_t[:, eos_idx] + 2.0  # Boost EOS probability
            
            # Fix: Penalize <unk> and <pad>
            unk_idx = self.vocab.tgt['<unk>']
            pad_idx = self.vocab.tgt['<pad>']
            log_p_t[:, unk_idx] = log_p_t[:, unk_idx] - 10.0
            log_p_t[:, pad_idx] = log_p_t[:, pad_idx] - 10.0
            
            max_elem, max_idx = torch.max(log_p_t, dim=1)
            next_word = self.vocab.tgt.id2word[max_idx.item()]

            # Fix: Skip invalid tokens
            if next_word in ['<pad>', ''] or not next_word.strip():
                continue

            # Fix: Prevent repetition - check if same token appears too frequently
            if len(hypothesis) > 5:
                recent_tokens = hypothesis[-5:]
                if recent_tokens.count(next_word) >= 3:  # Same token appears 3+ times in last 5
                    # Skip this token, try second best
                    log_p_t[0, max_idx.item()] = -float('inf')
                    max_elem, max_idx = torch.max(log_p_t, dim=1)
                    next_word = self.vocab.tgt.id2word[max_idx.item()]
                    if next_word in ['<pad>', ''] or not next_word.strip():
                        break

            # If the decoded word is "</s>", stop the inference and return the translated
            # sentence.
            if next_word == "</s>":
                # Allow </s> if: (1) meets min_length, OR (2) sequence is already quite long (>= 10 tokens)
                if actual_length >= min_length or actual_length >= 10:
                    break
                elif actual_length >= max_reasonable_length:
                    # Force stop if sequence is too long
                    break
                else:
                    # Skip </s> if too short, continue decoding
                    continue

            hypothesis.append(next_word)
            o_prev = o_t

        # Fix: Filter out invalid tokens before decoding
        decoded_tokens = [t for t in hypothesis[1:] if t not in ['<s>', '</s>', '<unk>', '<pad>'] and t.strip()]
        try:
            translated_sentence = self.sp.decode(decoded_tokens)
            translated_sentence = translated_sentence.replace('⁇', '').replace('', '').strip()
        except:
            translated_sentence = " ".join([t.replace('▁', '') if t.startswith('▁') else t for t in decoded_tokens])
    
        return decoded_tokens, translated_sentence

    @property
    def device(self):
        """device: Determine which device to place the Tensors upon, CPU or GPU."""
        # Fix: Handle different attention types
        if hasattr(self, 'att_projection'):
            return self.att_projection.weight.device
        elif hasattr(self, 'att_projection_enc'):
            return self.att_projection_enc.weight.device
        else:
            # Fallback to any parameter
            return next(self.parameters()).device

    @staticmethod
    def load(model_path):
        """Load the model from a file."""
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        kwargs = params["args"]
        # Fix: Provide default values for new parameters if not in saved model
        if "teacher_forcing_ratio" not in kwargs:
            kwargs["teacher_forcing_ratio"] = 0.5
        if "frequency_penalty" not in kwargs:
            kwargs["frequency_penalty"] = 0.5
        if "attention_type" not in kwargs:
            kwargs["attention_type"] = "dot-product"
        model = RnnNMT(vocab=params["vocab"], **kwargs)
        model.load_state_dict(params["state_dict"])
        # Re-compute frequency penalty tensor after loading
        model._precompute_frequency_penalty()
        return model

    def save(self, path):
        """Save the model to a file."""
        params = {
            "args": dict(word_embed_size=self.model_embeddings_source.word_embed_size,
                         hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         teacher_forcing_ratio=self.teacher_forcing_ratio,
                         frequency_penalty=self.frequency_penalty,
                         attention_type=self.attention_type),
            "vocab": self.vocab,
            "state_dict": self.state_dict()
        }
        torch.save(params, path)


    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        """Given a single source sentence, perform beam search, yielding translations in
        the target language.

        Args:
            src_sent (List[str]): a single source sentence (words)
            beam_size (int, optional): Beam size. Defaults to 5.
            max_decoding_time_step (int, optional): Maximum number of time steps to unroll
                the decoding RNN for. Defaults to 70.

        Returns:
            hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        # Fix: Use correct projection layer based on attention type
        if self.attention_type == 'dot-product' or self.attention_type == 'multiplicative':
            src_encodings_att_linear = self.att_projection(src_encodings)
        elif self.attention_type == 'additive':
            src_encodings_att_linear = self.att_projection_enc(src_encodings)
        else:
            # Fallback (should not reach here if attention_type is validated)
            src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']
        min_length = 3  # Minimum length for a valid translation (excluding <s> and </s>)

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        max_reasonable_length = 30  # Maximum reasonable length before forcing EOS
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            
            # Fix: Check if hypotheses is empty (all were skipped)
            if hyp_num == 0:
                break
            
            # Fix: Force EOS if sequences are getting too long (check at end of iteration)
            # This will be checked after processing current step

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                        src_encodings_att_linear.size(1),
                                                        src_encodings_att_linear.size(2))

            y_tm1 = self.vocab.tgt.to_input_tensor(list([hyp[-1]]
                                            for hyp in hypotheses), device=self.device)
            y_t_embed = self.model_embeddings_target(y_tm1)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings,
                                            exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)
            
            # Fix: Apply frequency penalty to suppress high-frequency words
            # Use pre-computed penalty tensor for much better performance
            if self.frequency_penalty_tensor is not None:
                # Apply pre-computed penalty tensor (vectorized, very fast!)
                log_p_t = log_p_t - self.frequency_penalty_tensor.unsqueeze(0).to(log_p_t.device)
            
            # Fix: Penalize </s> token if sequence is too short, but allow it after reasonable length
            if t < min_length + 1:  # Don't allow </s> in first min_length steps
                eos_idx = self.vocab.tgt['</s>']
                log_p_t[:, eos_idx] = log_p_t[:, eos_idx] - 50.0  # Large penalty for early </s>
            elif t >= max_reasonable_length:  # Encourage EOS if sequence is already long
                eos_idx = self.vocab.tgt['</s>']
                log_p_t[:, eos_idx] = log_p_t[:, eos_idx] + 2.0  # Boost EOS probability
            
            # Fix: Penalize <unk> token to reduce invalid generations
            unk_idx = self.vocab.tgt['<unk>']
            log_p_t[:, unk_idx] = log_p_t[:, unk_idx] - 10.0  # Penalty for <unk>

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode="trunc")
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                # Fix: Ensure hyp_word_id is valid and get the word
                if hyp_word_id not in self.vocab.tgt.id2word:
                    # Skip invalid token IDs - don't add to hypothesis
                    continue
                else:
                    hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                
                # Fix: Skip if word is invalid (empty, pad, or other control tokens)
                if hyp_word in ['<pad>', ''] or not hyp_word.strip():
                    continue

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    # Fix: Accept EOS token if sequence is long enough
                    # new_hyp_sent includes <s> and </s>, so length should be at least min_length + 2
                    actual_length = len(new_hyp_sent) - 2  # Excluding <s> and </s>
                    # Allow EOS if: (1) meets min_length, OR (2) sequence is already quite long (>= 10 tokens)
                    if actual_length >= min_length or actual_length >= 10:
                        # Add length penalty: prefer sequences closer to reference length
                        # Length penalty: (5 + len(hyp)) / (5 + 1) ^ alpha, where alpha = 0.6
                        # This encourages longer sequences but not too long
                        length_penalty = ((5.0 + actual_length) / 6.0) ** 0.6
                        adjusted_score = cand_new_hyp_score / length_penalty
                        completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                               score=adjusted_score))
                    else:
                        # If too short, continue decoding instead of completing
                        # But only if we haven't reached a reasonable maximum length
                        if actual_length < 20:  # Allow continuation if still reasonable
                            new_hypotheses.append(new_hyp_sent)
                            live_hyp_ids.append(prev_hyp_id)
                            # Large penalty for early </s> to discourage it
                            penalty = 20.0 * (min_length - actual_length)  # Larger penalty for shorter sequences
                            new_hyp_scores.append(cand_new_hyp_score - penalty)
                        else:
                            # Force completion if sequence is already long enough
                            length_penalty = ((5.0 + actual_length) / 6.0) ** 0.6
                            adjusted_score = cand_new_hyp_score / length_penalty
                            completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                                   score=adjusted_score))
                else:
                    # Fix: Prevent infinite loops by checking for repeated tokens
                    # Improved repetition detection: check for n-gram repetition
                    should_skip = False
                    
                    # Check 1: Simple token repetition in recent history
                    if len(new_hyp_sent) > 10:
                        last_10 = new_hyp_sent[-10:]
                        token_counts = {}
                        for tok in last_10:
                            token_counts[tok] = token_counts.get(tok, 0) + 1
                        max_count = max(token_counts.values()) if token_counts else 0
                        if max_count >= 8:  # Same token appears >=80% of the time in last 10
                            should_skip = True
                    
                    # Check 2: Bigram repetition (e.g., "the US" repeated)
                    if not should_skip and len(new_hyp_sent) > 6:
                        last_6 = new_hyp_sent[-6:]
                        bigrams = [(last_6[i], last_6[i+1]) for i in range(len(last_6)-1)]
                        bigram_counts = {}
                        for bg in bigrams:
                            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
                        max_bigram_count = max(bigram_counts.values()) if bigram_counts else 0
                        if max_bigram_count >= 3:  # Same bigram appears 3+ times in last 6 tokens
                            should_skip = True
                    
                    # Check 3: Diversity penalty - penalize if vocabulary is too small
                    if not should_skip and len(new_hyp_sent) > 5:
                        unique_tokens = len(set(new_hyp_sent[1:]))  # Exclude <s>
                        total_tokens = len(new_hyp_sent) - 1
                        diversity_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
                        if diversity_ratio < 0.3:  # Less than 30% unique tokens
                            # Apply diversity penalty instead of skipping
                            diversity_penalty = 5.0 * (0.3 - diversity_ratio)
                            cand_new_hyp_score -= diversity_penalty
                    
                    if should_skip:
                        continue
                    
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break
            
            # Fix: Check if no new hypotheses were added (all were skipped)
            if len(new_hypotheses) == 0:
                break
            
            # Fix: Force EOS if sequences are getting too long
            if t >= max_reasonable_length:
                # Check if any hypothesis is long enough to complete
                for i, hyp in enumerate(new_hypotheses):
                    actual_length = len(hyp) - 1  # Excluding <s>
                    if actual_length >= 5:  # At least 5 tokens, accept it
                        # Add EOS and complete
                        completed_hyp = hyp[1:] + ['</s>']  # Remove <s>, add </s>
                        length_penalty = ((5.0 + actual_length) / 6.0) ** 0.6
                        adjusted_score = new_hyp_scores[i] / length_penalty
                        completed_hypotheses.append(Hypothesis(value=completed_hyp[:-1],  # Remove </s> from value
                                                               score=adjusted_score))
                if len(completed_hypotheses) > 0:
                    break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        # Fix: Handle case where no hypotheses completed
        if len(completed_hypotheses) == 0:
            # Use the best incomplete hypothesis
            if len(hypotheses) > 0 and len(hypotheses[0]) > 1:
                # Remove <s> and any trailing invalid tokens
                best_hyp = hypotheses[0][1:]  # Remove <s>
                # Filter out <unk> and other invalid tokens if too many
                filtered_hyp = [tok for tok in best_hyp if tok not in ['<unk>', '<pad>']]
                if len(filtered_hyp) > 0:
                    completed_hypotheses.append(Hypothesis(value=filtered_hyp,
                                                           score=hyp_scores[0].item()))
                else:
                    # Fallback: return empty hypothesis
                    completed_hypotheses.append(Hypothesis(value=[],
                                                           score=hyp_scores[0].item()))
            else:
                # Complete fallback
                completed_hypotheses.append(Hypothesis(value=[],
                                                       score=0.0))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses


    