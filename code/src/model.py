import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter


class SubwordSegmentalLM(nn.Module):

    def __init__(self, char_vocab_size, lex_vocab_size, chars2lex,
                 lenlex_vocab_sizes, chars2lenlex,
                 input_size, hidden_size, feedforward_size, num_layers, num_heads, dropout,
                 max_seg_len, eom_id, pad_id, last_alpha_id, reg_exp, device, mixture_gate="sigmoid", mixture_temp=1.0,
                 seg_gen="mix", char_gen="char", encoder_type="lstm", decoder_type="lstm", context_len=0, vectorise=True,
                 lenlex=False, timeit=False, lenlex_parallel=False,
                 len_dist="learned", only_subword=True):
        super(SubwordSegmentalLM, self).__init__()
        self.char_vocab_size = char_vocab_size

        self.lex_vocab_size = lex_vocab_size
        if lenlex:
            self.lenlex_vocab_sizes = lenlex_vocab_sizes
            self.chars2lenlex = chars2lenlex

        else:
            self.chars2lex = chars2lex

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feedforward_size = feedforward_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.eom_id = eom_id
        self.pad_id = pad_id
        self.last_alpha_id = last_alpha_id
        self.reg_exp = reg_exp
        self.mixture_gate = mixture_gate
        self.mixture_temp = torch.tensor(mixture_temp)
        self.device = device
        self.seg_gen = seg_gen
        self.char_gen = char_gen
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.context_len = context_len

        self.vectorise = vectorise
        self.lenlex = lenlex
        self.timeit = timeit
        self.lenlex_parallel = lenlex_parallel
        self.len_dist = len_dist
        self.only_subword = only_subword

        self.embedding_drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=input_size, padding_idx=pad_id)
        self.embedding.weight.data.uniform_(-0.08, 0.08)
        self.embedding.weight.data[pad_id] = 0.0

        self.history_drop = nn.Dropout(dropout)

        self.history_drop = nn.Dropout(dropout)
        if encoder_type == "lstm":
            self.encoder = LSTMCharEncoder(char_vocab_size, input_size, hidden_size, num_layers, dropout)
        else:
            self.encoder = TransformerCharEncoder(char_vocab_size, input_size, feedforward_size, num_layers, dropout,
                                                  num_heads=num_heads)

        if decoder_type == "lstm":
            self.char_decoder = LSTMCharDecoder(char_vocab_size, input_size, hidden_size, num_layers, dropout)
        else:
            self.char_decoder = TransformerCharDecoder(char_vocab_size, input_size, feedforward_size, num_layers,
                                                       dropout, num_heads=1)

        if lenlex:
            if lenlex_parallel:
                self.lex_decoder = MultiLexDecoder(lenlex_vocab_sizes, hidden_size, num_layers, dropout, self.device)
            else:
                lex_decoders = []
                for seg_len in lenlex_vocab_sizes:
                    lex_decoders.append(LexDecoder(lenlex_vocab_sizes[seg_len], hidden_size, num_layers, dropout))
                self.lex_decoders = nn.ModuleList(lex_decoders)

            if self.len_dist == "learned":
                self.seg_len_model = SegLenModel(max_seg_len, hidden_size)

        else:
            self.lex_decoder = LexDecoder(lex_vocab_size, hidden_size, num_layers, dropout)

        if mixture_gate == "sigmoid":
            self.mixture_gate_func = nn.Linear(hidden_size, 1)
            for name, weights in self.mixture_gate_func.named_parameters():
                weights.data.uniform_(-0.08, 0.08)

        elif mixture_gate == "lstm":
            self.mixture_gate_func = LSTMGate(input_size)

    def forward(self, input_ids, hidden_state, target_ids, mode="forward"):
        # Encode histories
        extended_input_ids = torch.cat((input_ids, target_ids[-1].unsqueeze(0)), dim=0)
        input_embeddings = self.embedding(extended_input_ids)
        input_embeddings = self.embedding_drop(input_embeddings)

        bptt_len = input_ids.shape[0]
        batch_size = input_ids.shape[1]
        max_seg_len = min(self.max_seg_len, bptt_len)

        if self.encoder_type == "lstm":
            hidden_states, final_states = self.encoder(input_embeddings[0: -1], hidden_state)
        else:
            input_mask = self.encoder.generate_square_subsequent_mask(len(input_ids), context_len=self.context_len).to(self.device)
            hidden_states = self.encoder(input_embeddings[0: -1], input_mask)
        history_encodings = hidden_states * 0.5
        history_encodings = self.history_drop(history_encodings)

        # Compute log probabilities for all possible segments
        target_alphabet = torch.where(target_ids <= self.last_alpha_id, True, False)
        seg_embeddings = []
        seg_target_ids = []
        seg_lens = []

        for seg_end in range(max_seg_len - 1, bptt_len + max_seg_len-1):  # end of first possible seg to end of sequence
            seg_target_alphas = target_alphabet[seg_end - max_seg_len + 1: seg_end + 1]
            seg_lens.append([])
            for seq_num in range(batch_size):
                if not seg_target_alphas[0][seq_num]:
                    seg_len = 1
                else:
                    for j in range(len(seg_target_alphas)):
                        if seg_target_alphas[j][seq_num]:
                            seg_len = j+1
                        else:
                            break
                seg_lens[-1].append(seg_len)

            seg_embeddings.append(input_embeddings[seg_end - max_seg_len + 1: seg_end + 2].clone())
            seg_target_ids.append(target_ids[seg_end - max_seg_len + 1: seg_end + 1].clone())

            if seg_end >= bptt_len:  # pad with zero embeddings and pad ids
                seg_embeddings[-1] = torch.cat((seg_embeddings[-1], torch.zeros((seg_end - bptt_len + 1, batch_size,
                                                                                 input_embeddings.shape[-1]),
                                                                                device=self.device)))
                seg_target_ids[-1] = torch.cat((seg_target_ids[-1], torch.full((seg_end - bptt_len + 1, batch_size),
                                                                               fill_value=self.pad_id,
                                                                               device=self.device)))

        seg_embeddings = torch.stack(seg_embeddings, dim=1).view(max_seg_len + 1, -1, self.input_size)
        seg_hidden_states = history_encodings.view(1, -1, self.hidden_size)
        if self.decoder_type == "lstm":
            char_logits, _ = self.char_decoder(seg_embeddings, seg_hidden_states)
        else:
            input_mask = self.char_decoder.generate_square_subsequent_mask(max_seg_len + 1).to(self.device)
            char_logits = self.char_decoder(seg_embeddings, input_mask)

        full_char_logp = F.log_softmax(char_logits, dim=-1)
        target_prob_ids = torch.stack(seg_target_ids, dim=1).view(max_seg_len, -1)
        char_logp = torch.gather(full_char_logp[0: max_seg_len], dim=-1, index=target_prob_ids.unsqueeze(-1)).squeeze(-1)
        loginf = 1000000.0

        if self.lex_vocab_size > 0 or self.lenlex:

            if self.lenlex:

                if self.len_dist == "learned":
                    seg_len_logits = self.seg_len_model(hidden_states)
                    seg_len_logp = F.log_softmax(seg_len_logits, dim=-1)
                else:
                    seg_len_logp = torch.log(torch.tensor(1 / self.max_seg_len))

                if self.lenlex_parallel:
                    full_lenlex_logp = {}
                    lex_logits = self.lex_decoder(hidden_states)
                    full_lex_logp = F.log_softmax(lex_logits, dim=-1)

                    for seg_len in self.lenlex_vocab_sizes:
                        len_logp = seg_len_logp[:, :, seg_len - 1].unsqueeze(-1)
                        full_lenlex_logp[seg_len] = len_logp + full_lex_logp[seg_len-1]
                        full_lenlex_logp[seg_len] = torch.cat(
                             (full_lenlex_logp[seg_len], torch.full((full_lenlex_logp[seg_len].shape[0], full_lenlex_logp[seg_len].shape[1], 1),
                                                        fill_value=-loginf, device=self.device)), dim=-1)

                else:
                    full_lenlex_logp = {}
                    for seg_len in self.lenlex_vocab_sizes:
                        lex_logits = self.lex_decoders[seg_len-1](hidden_states)
                        full_lex_logp = F.log_softmax(lex_logits, dim=-1)
                        if self.len_dist == "learned":
                            len_logp = seg_len_logp[:, :, seg_len-1].unsqueeze(-1)
                        else:
                            len_logp = seg_len_logp
                        full_lex_logp = len_logp + full_lex_logp

                        full_lenlex_logp[seg_len] = torch.cat(
                            (full_lex_logp, torch.full((full_lex_logp.shape[0], full_lex_logp.shape[1], 1),
                                                       fill_value=-loginf, device=self.device)), dim=-1)

                lex_logp = {}
                for seg_len in range(1, max_seg_len + 1):
                    if seg_len == 1 and self.char_gen == "char":
                        continue

                    lex_logp[seg_len] = []

                    for seg_start in range(bptt_len - (seg_len - 1)):
                        seg_char_ids = target_ids[seg_start: seg_start + seg_len].T.tolist()
                        seg_char_ids = [tuple(char_ids) for char_ids in seg_char_ids]
                        seg_lex_ids = torch.LongTensor([self.chars2lenlex[seg_len][char_ids] if char_ids in self.chars2lenlex[seg_len]
                                                        else self.lenlex_vocab_sizes[seg_len]  # not in segment lexicon
                                                        for char_ids in seg_char_ids]).to(self.device)
                        lex_logp[seg_len].append(torch.gather(full_lenlex_logp[seg_len][seg_start], dim=-1,
                                                              index=seg_lex_ids.unsqueeze(-1)).squeeze(-1))
                    lex_logp[seg_len] = torch.stack(lex_logp[seg_len], dim=0)

            else:
                # Lexicon generation
                lex_logits = self.lex_decoder(hidden_states)
                full_lex_logp = F.log_softmax(lex_logits, dim=-1)
                full_lex_logp = torch.cat((full_lex_logp, torch.full((full_lex_logp.shape[0], full_lex_logp.shape[1], 1),
                                                                     fill_value=-loginf, device=self.device)), dim=-1)

                lex_logp = {}
                for seg_len in range(1, max_seg_len + 1):
                    if seg_len == 1 and self.char_gen == "char":
                        continue

                    lex_logp[seg_len] = []

                    for seg_start in range(bptt_len - (seg_len-1)):
                        seg_char_ids = target_ids[seg_start: seg_start+seg_len].T.tolist()
                        seg_char_ids = [tuple(char_ids) for char_ids in seg_char_ids]
                        seg_lex_ids = torch.LongTensor([self.chars2lex[char_ids] if char_ids in self.chars2lex
                                                        else self.lex_vocab_size  # not in segment lexicon
                                       for char_ids in seg_char_ids]).to(self.device)
                        lex_logp[seg_len].append(torch.gather(full_lex_logp[seg_start], dim=-1,
                                                              index=seg_lex_ids.unsqueeze(-1)).squeeze(-1))
                    lex_logp[seg_len] = torch.stack(lex_logp[seg_len], dim=0)

            # Compute log mixture cofficients
            logits = self.mixture_gate_func(hidden_states).squeeze(-1)
            logits = logits/self.mixture_temp

            log_char_proportions = F.logsigmoid(logits)
            log_lex_proportions = F.logsigmoid(-logits)

        seg_logp = {}
        for seg_len in range(1, max_seg_len + 1):
            end_batch_index = (bptt_len - (seg_len - 1)) * batch_size

            seg_logp[seg_len] = torch.sum(char_logp[0: seg_len, 0: end_batch_index], dim=0) \
                                + full_char_logp[seg_len, 0: end_batch_index, self.eom_id]
            seg_logp[seg_len] = seg_logp[seg_len].view(-1, batch_size)

            if self.only_subword:
                valid_segs = torch.tensor(seg_lens[0: bptt_len - (seg_len - 1)], device=self.device) >= seg_len
                seg_logp[seg_len] = torch.where(valid_segs, seg_logp[seg_len], torch.full_like(seg_logp[seg_len],
                                                                                          fill_value=-loginf))

            if self.lex_vocab_size > 0:
                # Calculate weighted average of character and lexical generation probabilities
                seg_log_char_proportions = log_char_proportions[0: bptt_len - (seg_len - 1)]
                seg_log_lex_proportions = log_lex_proportions[0: bptt_len - (seg_len - 1)]

                if seg_len < 2:

                    if self.char_gen == "char":
                        seg_logp[seg_len] = seg_log_char_proportions + seg_logp[seg_len]
                    elif self.char_gen == "lex":
                        neginf_log_proportions = torch.full_like(seg_log_lex_proportions, fill_value=-loginf)
                        seg_log_lex_proportions = torch.where(lex_logp[seg_len] > -loginf, seg_log_lex_proportions,
                                                              neginf_log_proportions)
                        seg_log_char_proportions = torch.where(seg_log_lex_proportions > -loginf,
                                                               torch.tensor(-loginf, device=self.device),
                                                               seg_log_char_proportions)

                        char_element = seg_log_char_proportions + seg_logp[seg_len]
                        lex_element = seg_log_lex_proportions + lex_logp[seg_len]
                        seg_logp[seg_len] = torch.logsumexp(torch.stack([char_element, lex_element]), dim=0)
                    elif self.char_gen == "mix":
                        neginf_log_proportions = torch.full_like(seg_log_lex_proportions, fill_value=-loginf)
                        seg_log_lex_proportions = torch.where(lex_logp[seg_len] > -loginf, seg_log_lex_proportions,
                                                              neginf_log_proportions)

                        char_element = seg_log_char_proportions + seg_logp[seg_len]
                        lex_element = seg_log_lex_proportions + lex_logp[seg_len]
                        seg_logp[seg_len] = torch.logsumexp(torch.stack([char_element, lex_element]), dim=0)

                else:
                    if self.seg_gen == "mix":

                        neginf_log_proportions = torch.full_like(seg_log_lex_proportions, fill_value=-loginf)
                        seg_log_lex_proportions = torch.where(lex_logp[seg_len] > -loginf, seg_log_lex_proportions,
                                                              neginf_log_proportions)

                        char_element = seg_log_char_proportions + seg_logp[seg_len]
                        lex_element = seg_log_lex_proportions + lex_logp[seg_len]
                        seg_logp[seg_len] = torch.logsumexp(torch.stack([char_element, lex_element]), dim=0)
                    elif self.seg_gen == "lex":
                        neginf_log_proportions = torch.full_like(seg_log_lex_proportions, fill_value=-loginf)
                        seg_log_lex_proportions = torch.where(lex_logp[seg_len] > -loginf, seg_log_lex_proportions,
                                                              neginf_log_proportions)
                        seg_log_char_proportions = torch.where(seg_log_lex_proportions > -loginf,
                                                               torch.tensor(-loginf, device=self.device),
                                                               seg_log_char_proportions)

                        char_element = seg_log_char_proportions + seg_logp[seg_len]
                        lex_element = seg_log_lex_proportions + lex_logp[seg_len]
                        seg_logp[seg_len] = torch.logsumexp(torch.stack([char_element, lex_element]), dim=0)

        if mode == "forward":
            # Compute alpha values and expected length factor
            log_alpha = torch.zeros((bptt_len + 1, batch_size), device=self.device)
            if self.reg_exp > 1:
                log_pv = torch.zeros((bptt_len + 1, batch_size), device=self.device)

            for t in range(1, bptt_len + 1):  # from alpha_1 to alpha_bptt_len
                alpha_sum_elements = []
                if self.reg_exp > 1:
                    pv_sum_elements = []

                range_j = list(range(max(0, t - max_seg_len), t))

                if not self.vectorise:
                    for j in range_j:
                        alpha_sum_elements.append((log_alpha[j] + seg_logp[t - j][j]))
                        if self.reg_exp > 1:  # Use expected length penalty
                            if j > 0:
                                log_p1v2 = log_alpha[j] + seg_logp[t - j][j] + \
                                           torch.log(torch.FloatTensor([(t - j)**self.reg_exp]).to(self.device))
                                log_p2v1 = seg_logp[t - j][j] + log_pv[j]
                                pv_sum_elements.append(torch.logsumexp(torch.stack([log_p1v2, log_p2v1]), dim=0))
                            else:  # First segment r value should be initialised to log p*v
                                log_p1v1 = seg_logp[t - j][j] + torch.log(torch.FloatTensor([(t - j)**self.reg_exp])
                                                                          .to(self.device))
                                pv_sum_elements.append(log_p1v1)
                    log_alpha[t] = torch.logsumexp(torch.stack(alpha_sum_elements), dim=0)

                    if self.reg_exp > 1:
                        log_pv[t] = torch.logsumexp(torch.stack(pv_sum_elements), dim=0)

                else:
                    log_alphas_t = log_alpha[range_j[0]: range_j[-1] + 1]
                    seg_logp_elements = []
                    regs_t = torch.zeros((len(range_j), 1)).to(self.device)

                    for j in range_j:
                        seg_logp_elements.append(seg_logp[t - j][j])
                        regs_t[j - range_j[0]] = torch.log(torch.FloatTensor([(t - j)**self.reg_exp]))

                    seg_logp_t = torch.stack(seg_logp_elements)
                    log_alpha[t] = torch.logsumexp(log_alphas_t + seg_logp_t, dim=0)

                    if self.reg_exp > 1:
                        log_p1v2_t = log_alphas_t + seg_logp_t + regs_t

                        mask = torch.zeros_like(seg_logp_t)
                        mask[0] = torch.tensor(-loginf)
                        log_p2v1_t = seg_logp_t + log_pv[range_j[0]: range_j[-1] + 1] + mask

                        pv_sum_elements = torch.logsumexp(torch.stack([log_p1v2_t, log_p2v1_t]), dim=0)
                        log_pv[t] = torch.logsumexp(pv_sum_elements, dim=0)

            if self.reg_exp > 1:
                log_R = log_pv[-1] - log_alpha[-1]
            else:
                log_R = -loginf

            if self.encoder_type == "lstm":
                return log_alpha[-1], log_R, (final_states[0].detach(), final_states[1].detach())
            else:
                return log_alpha[-1], log_R

        elif mode == "segment":
            # Compute alpha values and store backpointers
            assert batch_size == 1  # Can only segment one batch at a time
            bps = torch.zeros((max_seg_len, bptt_len+1), dtype=torch.long, device=self.device)
            max_logps = torch.full((max_seg_len, bptt_len + 1), fill_value=0.0, device=self.device)
            for t in range(1, bptt_len + 1):  # from alpha_1 to alpha_bptt_len
                for j in range(max(0, t - max_seg_len), t):
                    # The current segment starts at j and ends at t
                    # The backpointer will point to the segment ending at j-1
                    max_bp = max(1, j)  # Maximum possible length of segment ending at j-1

                    # Compute the probability of the most likely sequence ending with the segment j-t (length t-j)
                    # For this most likely sequence ending at j-1, what is the final segment length?
                    bps[t-j-1, t] = torch.argmax(max_logps[0: max_bp, j])

                    # What is the probability of the most likely sequence ending at t with last segment j-t?
                    max_logps[t-j-1, t] = torch.max(max_logps[0: max_bp, j]) + seg_logp[t - j][j]

            # Backtrack from final state of most likely path
            best_path = []
            k = torch.tensor(bptt_len)
            bp = torch.argmax(max_logps[:, bptt_len])

            while k > 0:
                best_path.insert(0, torch.tensor(k)-1)
                prev_bp = bp
                bp = bps[bp, k]
                k = k - (prev_bp + 1)
            split_indices = best_path

            if self.encoder_type == "lstm":
                return split_indices, (final_states[0].detach(), final_states[1].detach())
            else:
                return split_indices

        elif mode == "generate":
            hidden_states, final_states = self.encoder(input_embeddings, hidden_state)
            final_hidden_state = hidden_states[-1].unsqueeze(0)
            logits, _ = self.char_decoder(input_embeddings[-1].unsqueeze(0), final_hidden_state)
            return logits

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def init_encoder(self, encoder_type, embeddings, weights):
        if encoder_type == "lstm":
            self.embedding.weight = nn.Parameter(embeddings)
            self.encoder.lstm = weights


class LSTMCharEncoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super(LSTMCharEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, embedding, hidden_state):
        hidden_states, final_states = self.lstm(embedding, hidden_state)
        return hidden_states, (final_states[0].detach(), final_states[1].detach())

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def get_init_states(self, batch_size):
        return (self.h_init_state.expand(-1, batch_size, -1).contiguous(),
                self.c_init_state.expand(-1, batch_size, -1).contiguous())


class CharEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super(CharEncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, embedding, hidden_state):
        embedding = self.drop(embedding)
        hidden_states, final_states = self.lstm(embedding, hidden_state)
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits, hidden_states, (final_states[0].detach(), final_states[1].detach())

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class LSTMCharDecoder(nn.Module):
    """
    Character by character generation of a segment, conditioned on the sequence history.
    """
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super(LSTMCharDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        for name, weights in self.lstm.named_parameters():
            weights.data.uniform_(-0.08, 0.08)

        for name, weights in self.fc.named_parameters():
            weights.data.uniform_(-0.08, 0.08)

        for name, weights in self.transform.named_parameters():
            weights.data.uniform_(-0.08, 0.08)

    def forward(self, embedding, init_hidden_states):
        init_hidden_states = self.transform(init_hidden_states)
        init_cell_states = torch.zeros_like(init_hidden_states)
        hidden_states, final_states = self.lstm(embedding, (init_hidden_states, init_cell_states))
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits, (final_states[0].detach(), final_states[1].detach())

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class LexDecoder(nn.Module):
    """
    Once-off lexical generation of a segment, conditioned on the sequence history.
    """
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(LexDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

        for name, weights in self.fc.named_parameters():
            weights.data.uniform_(-0.08, 0.08)

        for name, weights in self.transform.named_parameters():
            weights.data.uniform_(-0.08, 0.08)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.drop(hidden_states)
        logits = self.fc(hidden_states)
        return logits


class MultiLexDecoder(nn.Module):

    def __init__(self, lex_vocab_sizes, hidden_size, num_layers, dropout, device):
        super(MultiLexDecoder, self).__init__()
        self.lex_vocab_sizes = lex_vocab_sizes
        self.max_vocab_size = max(lex_vocab_sizes.values())
        self.num_channels = len(lex_vocab_sizes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.weight = Parameter(torch.empty((self.num_channels, hidden_size, self.max_vocab_size)))
        self.bias = Parameter(torch.empty(self.num_channels, 1, self.max_vocab_size))
        self.reset_parameters(self.weight, self.bias)
        self.device = device

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, hidden_states):
        input = hidden_states.view((1, -1, hidden_states.shape[-1]))
        input = torch.cat(self.num_channels * [input])

        logits = torch.matmul(input, self.weight) + self.bias
        logits = logits.view(self.num_channels, hidden_states.shape[0], hidden_states.shape[1], self.max_vocab_size)

        mask = torch.ones(logits.shape).to(self.device)
        mask[0, :, :, self.lex_vocab_sizes[1]: ] = 0.0
        logits = logits * mask

        return logits


class SegLenModel(nn.Module):
    """
    Model segment length probability distribution. conditioned on the sequence history.
    """

    def __init__(self, max_seg_len, hidden_size):
        super(SegLenModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, max_seg_len)

    def forward(self, hidden_states):
        logits = self.fc(hidden_states)
        return logits


class TransformerCharEncoder(nn.Module):
    def __init__(self, vocab_size, input_size, feedforward_size, num_layers, dropout, num_heads):
        super(TransformerCharEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.feedforward_size = feedforward_size
        self.num_layers = num_layers

        encoder_layer = TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=feedforward_size,
                                                dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model=input_size, dropout=dropout)  # This drops out input embeddings

    def forward(self, embedding, input_mask):
        embedding = embedding * math.sqrt(self.input_size)
        embedding = self.pos_encoder(embedding)
        output_states = self.transformer_encoder(embedding, input_mask)
        return output_states

    def generate_square_subsequent_mask(self, seq_len, context_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)

        if context_len != 0 and context_len != seq_len:
            context_mask = (torch.triu(torch.ones(seq_len, seq_len), diagonal=-(context_len-1)) == 1)
            mask = torch.logical_and(mask, context_mask)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerCharDecoder(nn.Module):

    def __init__(self, vocab_size, input_size, feedforward_size, num_layers, dropout, num_heads):
        super(TransformerCharDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.feedforward_size = feedforward_size
        self.num_layers = num_layers
        encoder_layer = TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=feedforward_size, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model=input_size, dropout=dropout)  # This drops out input embeddings
        self.decoder = nn.Linear(in_features=input_size, out_features=vocab_size)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, embedding, input_mask):
        embedding = embedding * math.sqrt(self.input_size)
        embedding = self.pos_encoder(embedding)
        output_states = self.transformer_encoder(embedding, input_mask)
        output_states = self.drop(output_states)
        logits = self.decoder(output_states)
        return logits


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


class LSTMGate(nn.Module):
    """
    Compute mixture coefficient with the gates of an LSTM cell.
    """
    def __init__(self, hidden_size):
        super(LSTMGate, self).__init__()
        self.hidden_size = hidden_size

        # i_t
        self.A_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # f_t
        self.A_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        self.A_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.A_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        i_t = torch.sigmoid(h_t @ self.A_i + self.b_i)
        f_t = torch.sigmoid(h_t @ self.A_f + self.b_f)
        g_t = torch.tanh(h_t @ self.A_g + self.b_g)
        o_t = torch.sigmoid(h_t @ self.A_o + self.b_o)
        c_t = f_t * h_t + i_t * g_t
        gated_h_t = o_t * torch.tanh(c_t)
        logits = self.linear(gated_h_t)
        return logits