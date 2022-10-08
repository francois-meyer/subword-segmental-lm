import json
import operator
from model import SubwordSegmentalLM
from lstm import LSTM
from torchtext.legacy.data import Field
from torch import nn, optim
import numpy as np
import torch
import math
import time
import itertools
from collections import Counter
from tqdm import tqdm
from data import Corpus, LineCorpus
import nltk
from utils import save_vocab, load_vocab, save_model, load_model, LexiconVocab
import functools
from torch.cuda.amp import GradScaler

import argparse

EOS_TOKEN = "<eos>"
torch.manual_seed(0)
torch.set_printoptions(precision=10, profile="full")


def tokenize_chars(line):
    # Split into characters
    return list(line)


def tokenize_segs(line, max_seg_len, char_segs, non_alpha=False):
    # Split into all possible segments
    segs = []
    for n in range(1, max_seg_len+1):
        if n == 1 and not char_segs:
            continue

        chars = list(line)
        segs_n = nltk.ngrams(chars, n=n)
        segs_n = ["".join(seg) for seg in segs_n]

        if not non_alpha and n > 1:  # Discard segments with non-alphabetical characters
            segs_n = [seg for seg in segs_n if seg.isalpha() and len(seg) == n]
        else:
            segs_n = [seg for seg in segs_n if len(seg) == n]
        segs.extend(segs_n)
    return segs


def tokenize_line(line, max_seg_len):
    # Split into all possible segments
    line = line.strip()

    if len(line) <= max_seg_len:
        return [line]
    else:
        return []


def flatten_lex_vocab(lex_vocab, max_seg_len, max_per_len, non_alpha, max_len_threshold):
    if max_len_threshold:  # Determine count of max_per_len'th most frequent segment of length max_seg_len
        if non_alpha:
            max_segs = [seg for seg in lex_vocab.itos[3:] if len(seg) == max_seg_len]
        else:
            max_segs = [seg for seg in lex_vocab.itos[3:] if len(seg) == max_seg_len and seg.isalpha()]
        max_freqs = {}
        for seg in max_segs:
            max_freqs[seg] = lex_vocab.freqs[seg]

        max_freqs_sorted = dict(sorted(max_freqs.items(), key=operator.itemgetter(1), reverse=True))
        max_freqs_top = dict(itertools.islice(max_freqs_sorted.items(), max_per_len))
        threshold = max_freqs_top[list(max_freqs_top)[-1]]
        print("Longest segment threshold:", threshold)
    else:
        threshold = 0

    freqs = Counter()
    for n in range(1, max_seg_len+1):
        n_segs = [seg for seg in lex_vocab.itos[3:] if len(seg) == n]
        n_freqs = {}
        for seg in n_segs:
            n_freqs[seg] = lex_vocab.freqs[seg]

        n_freqs_sorted = dict(sorted(n_freqs.items(), key=operator.itemgetter(1),reverse=True))

        if threshold:
            n_freqs_top = {k: v for k, v in n_freqs_sorted.items() if v >= threshold}
        else:
            n_freqs_top = dict(itertools.islice(n_freqs_sorted.items(), max_per_len))
        dict.update(freqs, n_freqs_top)

    itos = lex_vocab.itos[0:3]
    itos.extend(list(freqs.keys()))
    stoi = {seg: index for index, seg in enumerate(itos)}

    lex_vocab.freqs = freqs
    lex_vocab.itos = itos
    lex_vocab.stoi = stoi


def generate_text(model, vocab, start=" Run", gen_len=30):
    # Generate some text

    with torch.no_grad():
        model.eval()
        chars = list(start)

        for i in range(gen_len):
            input_ids = torch.tensor([[vocab.stoi[ch]] for ch in chars])
            target_ids = input_ids[1: ]
            input_ids = input_ids[0: -1]
            state_h, state_c = model.init_states(1)
            logits = model(input_ids, (state_h, state_c), target_ids=target_ids, mode="generate")

            last_char_logits = logits[0][-1]
            p = torch.nn.functional.softmax(last_char_logits).detach().numpy()
            char_index = np.argmax(p)
            if vocab.itos[char_index] == EOS_TOKEN:
                chars.append("\n")
            else:
                chars.append(vocab.itos[char_index])

        print("".join(chars))


def train_epoch(model, train_corpus, char_vocab, epoch, device, log_interval,
                batch_size, bptt_len, optimizer, clip, reg_coef, scaler, timeit,
                scheduler, cur_steps, separate_lines):

    state_h, state_c = model.init_states(batch_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    total_loss = 0
    total_nll = 0
    total_R = 0
    for batch_num, example in enumerate(tqdm(train_corpus.iters(train_corpus, batch_size=batch_size, bptt_len=bptt_len,
                                                                device=device))):
        cur_steps += 1
        model.train()
        model.zero_grad()

        input_ids = example.text.to(device)
        target_ids = example.target.to(device)

        if scaler:  # Automatic mixed precision unimplemented
            pass

        else:
            if model.encoder_type == "lstm":
                if separate_lines:
                    state_h, state_c = model.encoder.get_init_states(batch_size)
                log_alpha, log_R, (state_h, state_c) = model(input_ids, (state_h, state_c), target_ids, mode="forward")
            else:
                log_alpha, log_R = model(input_ids, (state_h, state_c), target_ids, mode="forward")

            numel = torch.numel(target_ids) - (target_ids == char_vocab.stoi["<pad>"]).sum()

            nll = - torch.sum(log_alpha) / numel

            if model.reg_exp > 1:  # Add expected length regularisation
                R = torch.sum(torch.exp(log_R)) / numel
                loss = nll + reg_coef * R
            else:
                R = torch.tensor(0.0)
                loss = nll

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item()
        total_nll += nll.item()
        total_R += R.item()

        if batch_num % log_interval == 0 and batch_num > 0:
            cur_loss = total_loss / log_interval
            cur_nll = total_nll / log_interval
            cur_R = total_R / log_interval
            print("| epoch {:3d} | loss {:5.2f} | R {:5.2f} | nll {:5.2f} | ppl {:5.2f} | bpc {:5.2f} | ".format(epoch,
                cur_loss, cur_R, cur_nll, math.exp(cur_nll), cur_nll / math.log(2)))
            total_loss = 0
            total_nll = 0
            total_R = 0

    return cur_steps


def evaluate(model, char_vocab, eval_corpus, device, batch_size, bptt_len, reg_coef, separate_lines):

    model.eval()
    with torch.no_grad():

        state_h, state_c = model.init_states(batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        total_loss = 0
        total_nll = 0
        total_R = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_predicts = 0
        lex_coefs = []
        for batch_num, example in enumerate(tqdm(eval_corpus.iters(eval_corpus, batch_size=batch_size,
                                                                   bptt_len=bptt_len, device=device))):
            input_ids = example.text.to(device)
            target_ids = example.target.to(device)

            if model.encoder_type == "lstm":
                if separate_lines:
                    state_h, state_c = model.encoder.get_init_states(batch_size)
                log_alpha, log_R, (state_h, state_c), batch_lex_coefs = model(input_ids, (state_h, state_c), target_ids, mode="forward")
                batch_lex_coefs = torch.flatten(batch_lex_coefs)
                lex_coefs.append(batch_lex_coefs)
            else:
                log_alpha, log_R = model(input_ids, (state_h, state_c), target_ids, mode="forward")

            numel = torch.numel(target_ids) - (target_ids == char_vocab.stoi["<pad>"]).sum()
            nll = - torch.sum(log_alpha) / numel

            if model.reg_exp > 1:  # Add expected length regularisation
                R = torch.sum(torch.exp(log_R)) / numel
                loss = nll + reg_coef * R
            else:
                R = torch.tensor(0.0)
                loss = nll

            total_loss += numel * loss.item()
            total_nll += numel * nll.item()
            total_R += numel * R.item()
            total_predicts += numel

        print(total_predicts)
        return total_loss / total_predicts, total_R / total_predicts, total_nll / total_predicts, lex_coefs


def segment(model, char_vocab, eval_corpus, device, bptt_len, separate_lines, num_lines=1, segment_mode="segment"):

    model.eval()
    with torch.no_grad():

        state_h, state_c = model.init_states(1)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        text = ""
        split_indices = []
        cur_index = 0

        for batch_num, example in enumerate(tqdm(eval_corpus.iters(eval_corpus, batch_size=1, bptt_len=bptt_len,
                                                              device=device))):  # batch_size must be 1 for now
            input_ids = example.text.to(device)
            target_ids = example.target.to(device)

            if model.encoder_type == "lstm":
                if separate_lines:
                    state_h, state_c = model.encoder.get_init_states(batch_size=1)
                batch_split_indices, (state_h, state_c) = model(input_ids, (state_h, state_c), target_ids, mode=segment_mode)
            else:
                batch_split_indices = model(input_ids, (state_h, state_c), target_ids, mode=segment_mode)
            batch_split_indices = [index.item() for index in batch_split_indices]
            batch_text = "".join([char_vocab.itos[id] for id in target_ids.t()[0]]).replace("<eos>", "\n").replace("<pad>", "\n")

            print(split_text(batch_text, batch_split_indices))
            batch_split_indices = [cur_index + index for index in batch_split_indices]
            split_indices.extend(batch_split_indices)
            text += batch_text

            cur_index += len(input_ids)
            if batch_num + 1 == num_lines:
                break

        split = split_text(text, split_indices)
        if split[-1] == "|":
            split = split[0: -1]

        print(text)
        print("------------------------------------------------")
        print(split)

        return split_indices


def split_text(text, split_indices):
    for counter, index in enumerate(split_indices):
        text = text[:index + counter + 1] + "-" + text[index + counter + 1:]
    return text


def train_model(model, char_vocab, train_corpus, valid_corpus, num_epochs, batch_size, bptt_len, clip,
                optimizer, scheduler, reg_coef, log_interval, model_path, auto_mixed_prec, timeit,
                last_model_path, optim_state_path, early_stop, separate_lines):
    start = time.time()

    # Initialise model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    model = model.to(device)
    print("Initialised %s model" % model.__class__.__name__)

    # Train model
    best_loss = float("inf")
    best_nll = float("inf")
    best_epoch = None
    if auto_mixed_prec:
        scaler = GradScaler()
    else:
        scaler = None

    cur_steps = 0
    for epoch in range(1, num_epochs+1):
        cur_steps = train_epoch(model, train_corpus, char_vocab, epoch, device, log_interval, batch_size, bptt_len, optimizer,
                        clip, reg_coef, scaler, timeit, scheduler, cur_steps, separate_lines)
        val_loss, val_R, val_nll = evaluate(model, char_vocab, valid_corpus, device, batch_size, bptt_len, reg_coef, separate_lines)
        print("| epoch {:3d} | valid loss {:5.2f} | valid R {:5.2f} | valid nll {:5.2f} | valid ppl {:5.2f} | "
              "valid bpc {:5.2f} | ".format(epoch, val_loss, val_R, val_nll, math.exp(val_nll), val_nll / math.log(2)))
        segment(model, char_vocab, valid_corpus, device, bptt_len, separate_lines, num_lines=10)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            save_model(model, model_path)
            best_loss = val_loss
            best_nll = val_nll
            best_epoch = epoch

        if epoch - best_epoch >= early_stop:  # Early stopping
            print("valid bpc not improved in %d epochs, stopping training early" % early_stop)
            break

        # Save current model and optimizer
        save_model(model, last_model_path)
        print("Last model saved at %s" % last_model_path)
        optim_state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        torch.save(optim_state, optim_state_path)
        print("Optimizer and scheduler states saved at %s" % optim_state_path)

    end = time.time()
    print("Training completed in %fs." % (end - start))
    print("Best model saved at %s" % model_path)
    print("Best model checkpoint:")

    best_ppl = math.exp(best_nll)
    best_bpc = best_nll / math.log(2)
    print("| epoch {:3d} | valid loss {:5.2f} | valid ppl {:5.2f} | valid bpc {:5.2f} | ".format(best_epoch, best_loss,
                                                                                                 best_ppl, best_bpc))

    valid_result = {"epoch": best_epoch, "ppl": best_ppl, "bpc": best_bpc}
    optim_state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    return valid_result, model, optim_state


def map_chars2lex(char_vocab, lex_vocab):
    chars2lex = {}
    for lex_id, seg in enumerate(lex_vocab.itos):
        if not seg.isalpha():
            continue
        seg_chars = tuple(seg)
        seg_char_ids = tuple((char_vocab.stoi[char] for char in seg_chars))
        chars2lex[seg_char_ids] = lex_id
    return chars2lex


def main():

    data_dir = "../data/"
    repo_dir = "../sslm/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=data_dir + "train.mock")
    parser.add_argument("--valid_path", default=data_dir + "valid.mock")

    parser.add_argument("--model_path", default=repo_dir + "output/sslm.model")
    parser.add_argument("--char_vocab_path", default=repo_dir + "output/char.vocab")
    parser.add_argument("--lex_vocab_path", default=repo_dir + "output/lex.vocab")
    parser.add_argument("--len_lex_vocab_path", default=repo_dir + "output/len_lex.vocab")
    parser.add_argument("--output_path", default=repo_dir + "output/result.txt")
    parser.add_argument("--last_model_path", default=repo_dir + "output/last_lm.model")
    parser.add_argument("--optim_state_path", default=repo_dir + "output/optim.state")
    parser.add_argument("--train_mode", default="start")
    parser.add_argument("--charlm_path", default="output/last_lm.model")
    parser.add_argument("--pretrained_char_vocab_path", default="output/char_lm.vocab")

    default_params = '{"num_epochs": 100,' \
                     '"max_seg_len": 5,' \
                     '"lexicon": true,' \
                     '"lex_min_count": 1,' \
                     '"lex_max_size": 10000,' \
                     '"char_gen": "mix",' \
                     '"seg_gen": "mix",' \
                     '"reg_exp": 1,' \
                     '"reg_coef": 5.0e-5,' \
                     '"mixture_gate": "sigmoid",' \
                     '"mixture_temp": 1.0,' \
                     '"input_size": 512,' \
                     '"hidden_size": 1024,' \
                     '"feedforward_size": 1024,' \
                     '"num_layers": 3,' \
                     '"num_heads": 4,' \
                     '"dropout": 0.2,' \
                     '"bptt_len": 120,' \
                     '"lr": 0.001,' \
                     '"batch_size": 64,' \
                     '"weight_decay": 1e-5,' \
                     '"lr_patience": 3,' \
                     '"early_stop": 6,' \
                     '"clip": 1.0,' \
                     '"log_interval": 10,' \
                     '"auto_mixed_prec": false,' \
                     '"encoder_type": "lstm",' \
                     '"decoder_type": "lstm",' \
                     '"context_len": 0,' \
                     '"max_per_len": 1000,' \
                     '"vectorise": true,' \
                     '"lenlex": true,' \
                     '"timeit": false,' \
                     '"lenlex_parallel": false,' \
                     '"len_dist": "learned",' \
                     '"lex_non_alpha": false,' \
                     '"max_len_threshold": false, ' \
                     '"load_lex": false, ' \
                     '"lex_file_path": "", ' \
                     '"load_encoder": false, ' \
                     '"only_subword": true}'

    parser.add_argument("--params", default=default_params)
    args = parser.parse_args()
    train_path = args.train_path
    valid_path = args.valid_path
    model_path = args.model_path
    char_vocab_path = args.char_vocab_path
    lex_vocab_path = args.lex_vocab_path
    len_lex_vocab_path = args.len_lex_vocab_path
    output_path = args.output_path
    last_model_path = args.last_model_path
    optim_state_path = args.optim_state_path
    train_mode = args.train_mode
    charlm_path = args.charlm_path
    pretrained_char_vocab_path = args.pretrained_char_vocab_path
    params = args.params

    params = json.loads(params)
    default_params = json.loads(default_params)

    for key in default_params:
        if key not in params:
            params[key] = default_params[key]

    train_path = params["train_path"] if "train_path" in params else train_path
    num_epochs = params["num_epochs"] if "num_epochs" in params else default_params["num_epochs"]
    max_seg_len = params["max_seg_len"] if "max_seg_len" in params else default_params["max_seg_len"]
    lexicon = params["lexicon"] if "lexicon" in params else default_params["lexicon"]
    lex_min_count = params["lex_min_count"] if "lex_min_count" in params else default_params["lex_min_count"]
    lex_max_size = params["lex_max_size"] if "lex_max_size" in params else default_params["lex_max_size"]
    char_gen = params["char_gen"] if "char_gen" in params else default_params["char_gen"]
    seg_gen = params["seg_gen"] if "seg_gen" in params else default_params["seg_gen"]
    reg_exp = params["reg_exp"] if "reg_exp" in params else default_params["reg_exp"]
    reg_coef = params["reg_coef"] if "reg_coef" in params else default_params["reg_coef"]
    mixture_gate = params["mixture_gate"] if "mixture_gate" in params else default_params["mixture_gate"]
    mixture_temp = params["mixture_temp"] if "mixture_temp" in params else default_params["mixture_temp"]
    input_size = params["input_size"] if "input_size" in params else default_params["input_size"]
    hidden_size = params["hidden_size"] if "hidden_size" in params else default_params["hidden_size"]
    feedforward_size = params["feedforward_size"] if "feedforward_size" in params else default_params["feedforward_size"]
    num_layers = params["num_layers"] if "num_layers" in params else default_params["num_layers"]
    num_heads = params["num_heads"] if "num_heads" in params else default_params["num_heads"]
    dropout = params["dropout"] if "dropout" in params else default_params["dropout"]
    bptt_len = params["bptt_len"] if "bptt_len" in params else default_params["bptt_len"]
    lr = params["lr"] if "lr" in params else default_params["lr"]
    batch_size = params["batch_size"] if "batch_size" in params else default_params["batch_size"]
    lr_patience = params["lr_patience"] if "lr_patience" in params else default_params["lr_patience"]
    lr_factor = params["lr_factor"] if "lr_factor" in params else default_params["lr_factor"]
    early_stop = params["early_stop"] if "early_stop" in params else default_params["early_stop"]
    clip = params["clip"] if "clip" in params else default_params["clip"]
    weight_decay = params["weight_decay"] if "weight_decay" in params else default_params["weight_decay"]
    log_interval = params["log_interval"] if "log_interval" in params else default_params["log_interval"]
    auto_mixed_prec = params["auto_mixed_prec"] if "auto_mixed_prec" in params else default_params["auto_mixed_prec"]
    encoder_type = params["encoder_type"] if "encoder_type" in params else default_params["encoder_type"]
    decoder_type = params["decoder_type"] if "decoder_type" in params else default_params["decoder_type"]
    context_len = params["context_len"] if "context_len" in params else default_params["context_len"]
    vectorise = params["vectorise"] if "vectorise" in params else default_params["vectorise"]
    max_per_len = params["max_per_len"] if "max_per_len" in params else default_params["max_per_len"]
    lenlex = params["lenlex"] if "lenlex" in params else default_params["lenlex"]
    timeit = params["timeit"] if "timeit" in params else default_params["timeit"]
    lex_non_alpha = params["lex_non_alpha"] if "lex_non_alpha" in params else default_params["lex_non_alpha"]
    len_dist = params["len_dist"] if "len_dist" in params else default_params["len_dist"]
    only_subword = params["only_subword"] if "only_subword" in params else default_params["only_subword"]
    lenlex_parallel = params["lenlex_parallel"] if "lenlex_parallel" in params else default_params["lenlex_parallel"]
    max_len_threshold = params["max_len_threshold"] if "max_len_threshold" in params else default_params["max_len_threshold"]
    load_lex = params["load_lex"] if "load_lex" in params else default_params["load_lex"]
    lex_file_path = params["lex_file_path"] if "lex_file_path" in params else default_params["lex_file_path"]

    load_encoder = params["load_encoder"] if "load_encoder" in params else default_params["load_encoder"]
    separate_lines = params["separate_lines"] if "separate_lines" in params else default_params["separate_lines"]

    # Load data
    print("Setting up training corpus...", end="")
    eos_token = "<eos>"  # end of sequence token
    CHAR_TEXT = Field(sequential=True, tokenize=tokenize_chars, use_vocab=True, eos_token=eos_token)
    encoding = "utf-8"
    if not separate_lines:
        train_corpus = Corpus(path=train_path, text_field=CHAR_TEXT, newline_eos=True, encoding=encoding)
        valid_corpus = Corpus(path=valid_path, text_field=CHAR_TEXT, newline_eos=True, encoding=encoding)
    else:
        train_corpus = LineCorpus(path=train_path, text_field=CHAR_TEXT, newline_eos=True, encoding=encoding)
        valid_corpus = LineCorpus(path=valid_path, text_field=CHAR_TEXT, newline_eos=True, encoding=encoding)
    print("DONE")

    if load_encoder:
        print("Loading char vocab...", end="")
        char_vocab = load_vocab(pretrained_char_vocab_path)
        char_vocab.itos.append("<eom>")
        char_vocab.stoi["<eom>"] = len(char_vocab.itos) - 1
        CHAR_TEXT.vocab = char_vocab

        original_stoi = dict(char_vocab.stoi)
    else:
        print("Building character vocab...", end="")
        specials = ["<eom>"]  # Add special end-of-morpheme (subword) symbol
        if separate_lines:  # Add special start-of-sequence symbol
            specials.append("<sos>")

        CHAR_TEXT.build_vocab(train_corpus, min_freq=1, specials=specials)
        char_vocab = CHAR_TEXT.vocab

    # Reorder char vocab - used for segmenting only subwords
    alpha_chars = []
    non_alpha_chars = []
    for char in char_vocab.itos:
        if char.isalpha():
            alpha_chars.append(char)
        else:
            non_alpha_chars.append(char)

    char_vocab.itos = alpha_chars + non_alpha_chars
    for index, char in enumerate(char_vocab.itos):
        char_vocab.stoi[char] = index
    last_alpha_id = len([char for char in char_vocab.itos if char.isalpha()]) - 1

    if load_encoder:
        # Find new indices for pretrained embeddings, based on vocab reordering
        original_indices = torch.tensor([original_stoi[char] for char in char_vocab.itos])

    print("DONE")
    char_vocab_size = len(char_vocab.itos)
    print("Character vocab size: %d" % char_vocab_size)
    save_vocab(char_vocab, char_vocab_path)

    if lexicon:
        print("Setting up lexicon vocab...", end="")
        if load_lex:
            LEX_TEXT = Field(sequential=True, tokenize=functools.partial(tokenize_line, max_seg_len=max_seg_len),
                             use_vocab=True, eos_token=eos_token)
            temp_train_corpus = Corpus(path=lex_file_path, text_field=LEX_TEXT, newline_eos=True, encoding=encoding)

            LEX_TEXT.build_vocab(temp_train_corpus, min_freq=1, max_size=None)
            lex_vocab = LEX_TEXT.vocab

            # Lexical vocab distribution
            counts = {l: 0 for l in range(1, max_seg_len + 1)}
            for item in lex_vocab.itos:
                counts[len(item)] += 1
            print("Raw vocab:", counts)

        else:
            char_segs = char_gen == "mix" or char_gen == "lex"
            LEX_TEXT = Field(sequential=True, tokenize=functools.partial(tokenize_segs, max_seg_len=max_seg_len,
                             char_segs=char_segs, non_alpha=lex_non_alpha), use_vocab=True, eos_token=eos_token)
            temp_train_corpus = Corpus(path=train_path, text_field=LEX_TEXT, newline_eos=True, encoding=encoding)

            if max_per_len > 0:  # Clip lexical vocab per segment length
                LEX_TEXT.build_vocab(temp_train_corpus, min_freq=lex_min_count, max_size=None)
                lex_vocab = LEX_TEXT.vocab

                # Lexical vocab distribution
                counts = {l: 0 for l in range(1, max_seg_len + 1)}
                for item in lex_vocab.itos:
                    counts[len(item)] += 1
                print("Raw vocab:", counts)
                flatten_lex_vocab(lex_vocab, max_seg_len, max_per_len=max_per_len, non_alpha=lex_non_alpha, max_len_threshold=max_len_threshold)

                # Lexical vocab distribution
                counts = {l: 0 for l in range(1, max_seg_len + 1)}
                for item in lex_vocab.itos:
                    counts[len(item)] += 1
                print("Flattened vocab:", counts)
            else:  # Clip lexical vocab overall
                LEX_TEXT.build_vocab(temp_train_corpus, min_freq=lex_min_count, max_size=lex_max_size)
                lex_vocab = LEX_TEXT.vocab

        del lex_vocab.itos[lex_vocab.stoi["<eos>"]]
        del lex_vocab.stoi["<eos>"]
        lex_vocab.stoi = {seg: index for index, seg in enumerate(lex_vocab.itos)}

        lex_vocab_size = len(lex_vocab.itos)
        save_vocab(lex_vocab, lex_vocab_path)
        chars2lex = map_chars2lex(char_vocab, lex_vocab)

        print("DONE")
        print("Lexical vocab size: %d" % lex_vocab_size)

        if lenlex:
            print("Modelling segment length distribution")
            print("Splitting lexicon by segment length")
            lenlex_vocab = LexiconVocab(lex_vocab, max_seg_len, non_alpha=lex_non_alpha)
            save_vocab(lenlex_vocab, len_lex_vocab_path)
            lenlex_vocab_sizes = {}
            chars2lenlex = {}
            for seg_len in lenlex_vocab.lens:
                lenlex_vocab_sizes[seg_len] = len(lenlex_vocab.lens[seg_len].itos)
                chars2lenlex[seg_len] = map_chars2lex(char_vocab, lenlex_vocab.lens[seg_len])

            print("DONE")
            print("Separate lexicon sizes:")
            for seg_len in lenlex_vocab.lens:
                print("%d: %d" % (seg_len, lenlex_vocab_sizes[seg_len]))
        else:
            lenlex_vocab_sizes = None
            chars2lenlex = None

    else:
        lex_vocab_size = 0
        chars2lex = None
        lenlex_vocab_sizes = None
        chars2lenlex = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SubwordSegmentalLM(char_vocab_size, lex_vocab_size, chars2lex,
                              lenlex_vocab_sizes, chars2lenlex,
                              input_size, hidden_size, feedforward_size, num_layers, num_heads, dropout,
                              max_seg_len=max_seg_len, eom_id=char_vocab.stoi["<eom>"], pad_id=char_vocab.stoi["<pad>"],
                              last_alpha_id=last_alpha_id, reg_exp=reg_exp, mixture_gate=mixture_gate, mixture_temp=mixture_temp,
                              device=device, char_gen=char_gen, seg_gen=seg_gen, encoder_type=encoder_type, decoder_type=decoder_type,
                              context_len=context_len, vectorise=vectorise, lenlex=lenlex, timeit=timeit,
                              lenlex_parallel=lenlex_parallel, len_dist=len_dist, only_subword=only_subword)

    if load_encoder:
        print("Loading pretrained encoder...")
        loaded = torch.load(charlm_path, map_location=device)

        # Reorder embeddings to correspond to vocab
        char_embeddings = loaded["embedding.weight"].to(device)
        eom_init = torch.zeros_like(char_embeddings[0], device=device)
        eom_init = nn.init.normal_(eom_init)
        char_embeddings = torch.vstack([char_embeddings, eom_init.unsqueeze(0)])
        char_embeddings = torch.index_select(char_embeddings, dim=0, index=original_indices.to(device))

        if encoder_type == "lstm":
            loaded_lstm = LSTM(char_vocab_size-1, input_size, hidden_size, num_layers, dropout)
            loaded_lstm.load_state_dict(torch.load(charlm_path, map_location=device))
            encoder_weights = loaded_lstm.lstm

        model.init_encoder(encoder_type=encoder_type, embeddings=char_embeddings, weights=encoder_weights)
        print("DONE. Initialised model encoder with pretrained weights.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, verbose=True, factor=lr_factor)

    if train_mode == "continue":
        print("Continuing training where it left off...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model = load_model(model=model, model_path=last_model_path, map_location=device)

        optim_state = torch.load(optim_state_path)
        optimizer.load_state_dict(optim_state["optimizer"])
        scheduler.load_state_dict(optim_state["scheduler"])

    print("Training model with params = %s " % json.dumps(params))
    valid_result, model, optim_state = train_model(model, char_vocab, train_corpus, valid_corpus, num_epochs,
                                                   batch_size, bptt_len, clip, optimizer, scheduler, reg_coef,
                                                   log_interval, model_path, auto_mixed_prec, timeit,
                                                   last_model_path, optim_state_path, early_stop, separate_lines)

    print("In case of continued training:")
    save_model(model, last_model_path)
    print("Last model saved at %s" % last_model_path)
    torch.save(optim_state, optim_state_path)
    print("Optimizer and scheduler states saved at %s" % optim_state_path)

    # Store validation performance for grid search
    open(output_path, 'w').close()
    with open(output_path, 'a') as output_file:
        output_file.write(json.dumps(params) + " " + json.dumps(valid_result))


if __name__ == "__main__":
    main()
