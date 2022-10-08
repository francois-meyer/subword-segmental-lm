from utils import load_model, load_vocab
import argparse
import torch
from model import SubwordSegmentalLM
from train import map_chars2lex, tokenize_chars, segment
from data import Corpus
from torchtext.legacy.data import Field
import os
import re


def main():
    reg_exp = 0
    max_seg_len = 5
    only_subword = True
    len_dist = "learned"
    hidden_size = 1024
    feedforward_size = 1024
    bptt_len = 120
    num_heads = 8
    num_layers = 3
    encoder_type = "lstm"
    lexicon = True
    language = "xhosa"
    lang = "xh"

    OUTPUT_DIR = "output/"
    MORPH_DATA_DIR = "morph/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=OUTPUT_DIR + "sslm." + lang)
    parser.add_argument("--char_vocab_path", default=OUTPUT_DIR + "char_vocab." + lang)
    parser.add_argument("--lex_vocab_path", default=OUTPUT_DIR + "lex_vocab." + lang)
    parser.add_argument("--lenlex_vocab_path", default=OUTPUT_DIR + "len_lex_vocab." + lang)
    parser.add_argument("--eval_corpus_path", default=MORPH_DATA_DIR + language + ".test.corpus")
    parser.add_argument("--eval_data_path", default=MORPH_DATA_DIR + language + ".clean.test.conll")
    parser.add_argument("--output_path", default=MORPH_DATA_DIR + "output.txt")
    parser.add_argument("--grid_path", default="grid.conf")

    args = parser.parse_args()
    model_path = args.model_path
    char_vocab_path = args.char_vocab_path
    lex_vocab_path = args.lex_vocab_path
    lenlex_vocab_path = args.lenlex_vocab_path
    eval_corpus_path = args.eval_corpus_path
    eval_data_path = args.eval_data_path
    output_path = args.output_path

    segment_mode = "segment"
    char_vocab = load_vocab(char_vocab_path)
    char_vocab.unk_index = char_vocab.stoi[","]

    if lexicon:
        lex_vocab = load_vocab(lex_vocab_path)
        lex_vocab_size = len(lex_vocab.itos)
        chars2lex = map_chars2lex(char_vocab, lex_vocab)
    else:
        lex_vocab_size = 0
        chars2lex = None

    lenlex = os.path.isfile(lenlex_vocab_path)
    if lenlex:
        lenlex_vocab = load_vocab(lenlex_vocab_path)
        lenlex_vocab_sizes = {}
        chars2lenlex = {}
        for seg_len in lenlex_vocab.lens:
            lenlex_vocab_sizes[seg_len] = len(lenlex_vocab.lens[seg_len].itos)
            chars2lenlex[seg_len] = map_chars2lex(char_vocab, lenlex_vocab.lens[seg_len])
            print(seg_len, lenlex_vocab.lens[seg_len].itos)
    else:
        lenlex_vocab_sizes = None
        chars2lenlex = None

    device = torch.device("cpu")

    params_dict = {}

    # Load model
    loaded = torch.load(model_path, map_location=device)
    input_size = loaded["embedding.weight"].shape[1]
    last_alpha_id = len([char for char in char_vocab.itos if char.isalpha()]) - 1

    params_dict["max_seg_len"] = max_seg_len
    params_dict["reg_exp"] = reg_exp
    params_dict["bptt_len"] = bptt_len
    params_dict["char_gen"] = "mix"
    params_dict["seg_gen"] = "mix"

    params = {"char_vocab_size": len(char_vocab.itos), "lex_vocab_size": lex_vocab_size,
              "chars2lex": chars2lex,
              "lenlex_vocab_sizes": lenlex_vocab_sizes, "chars2lenlex": chars2lenlex,
              "input_size": input_size, "hidden_size": hidden_size, "feedforward_size": feedforward_size,
              "num_layers": num_layers, "num_heads": num_heads,
              "dropout": 0.9, "max_seg_len": params_dict["max_seg_len"], "eom_id": char_vocab.stoi["<eom>"], "pad_id": char_vocab.stoi["<pad>"],
              "last_alpha_id": last_alpha_id, "reg_exp": params_dict["reg_exp"], "device": device,
              "char_gen": params_dict["char_gen"], "seg_gen": params_dict["seg_gen"],
              "lenlex": lenlex, "only_subword": only_subword, "len_dist": len_dist,
              "encoder_type": encoder_type}

    model = SubwordSegmentalLM(**params)
    model = load_model(model=model, model_path=model_path, map_location=device)

    TEXT = Field(sequential=True, tokenize=tokenize_chars, use_vocab=True, eos_token="<eos>")
    encoding = "utf-8"
    eval_corpus = Corpus(path=eval_corpus_path, text_field=TEXT, newline_eos=True, encoding=encoding)
    TEXT.vocab = char_vocab
    segment(model, char_vocab, eval_corpus, device, bptt_len=params_dict["bptt_len"], num_lines=1, segment_mode=segment_mode, separate_lines=False)
    evaluate(model, char_vocab, eval_corpus, bptt_len=params_dict["bptt_len"], eval_data_path=eval_data_path, output_path=output_path, device=device, segment_mode=segment_mode)


def get_split_indices(model, char_vocab, eval_corpus, bptt_len, device, segment_mode):
    model.eval()
    with torch.no_grad():

        state_h, state_c = model.init_states(1)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        text = ""
        split_indices = []
        cur_index = 0

        last_target_ids = None
        last_input_ids = None
        for batch_num, example in enumerate(eval_corpus.iters(eval_corpus, batch_size=1, bptt_len=bptt_len,
                                                                   device=device)):  # batch_size must be 1 for now
            input_ids = example.text.to(device)
            target_ids = example.target.to(device)

            if last_target_ids is not None:
                target_ids = torch.cat((last_target_ids, target_ids), dim=0).to(device)
                input_ids = torch.cat((last_input_ids, input_ids), dim=0).to(device)
                last_target_ids = None
                last_input_ids = None

            batch_len = len(target_ids)
            for i in range(len(target_ids) - 1, -1, -1):
                if not char_vocab.itos[target_ids[i]].isalpha():
                    batch_len = i + 1
                    break

            if batch_len < len(target_ids):
                last_target_ids = target_ids[batch_len: ]
                last_input_ids = input_ids[batch_len: ]
                target_ids = target_ids[0: batch_len]
                input_ids = input_ids[0: batch_len]

            if model.encoder_type == "lstm":
                batch_split_indices, (state_h, state_c) = model(input_ids, (state_h, state_c), target_ids, mode=segment_mode)
            else:
                batch_split_indices = model(input_ids, (state_h, state_c), target_ids, mode=segment_mode)

            batch_split_indices = [index.item() for index in batch_split_indices]
            batch_text = "".join([char_vocab.itos[id] for id in target_ids.t()[0]]).replace("<eos>", "\n")

            batch_split_indices = [cur_index + index for index in batch_split_indices]
            split_indices.extend(batch_split_indices)
            text += batch_text
            cur_index += len(input_ids)

    return text, split_indices


def get_eval_tokens(input_path):
    tokens = []
    with open(input_path) as file:
        for i, line in enumerate(file):
            tokens.append(line[0: line.index("|")-1])
    return tokens


def strip_punc(text):
    while not text[0].isalnum() and not text[0] == "%":
        text = text[1:]
        if len(text) == 0:
            return ""
    while not text[-1].isalnum() and not text[-1] == "%":
        text = text[:-1]
        if len(text) == 0:
            return ""
    return text


def split_text(text, split_indices):
    # Inserts character to indicate segmentation AFTER split indices
    # i.e. split_indices contains characters that END segments
    for counter, index in enumerate(split_indices):
        text = text[:index + counter + 1] + "|" + text[index + counter + 1:]
    return text


def split(text, split_indices):
    split_indices = [index for index in split_indices if index >= 0]

    space_indices = [i for i, ch in enumerate(text) if ch.isspace()]
    split_indices = [index for index in split_indices if index not in space_indices]

    split_tokens = []
    i = 0
    token_start = 0
    for token in text.split():
        token_end = token_start + len(token)
        token_splits = []
        while i < len(split_indices) and token_start <= split_indices[i] < token_end:
            token_splits.append(split_indices[i] - token_start)
            i += 1

        split_token = split_text(token, token_splits) if len(token_splits) > 0 else token
        split_token = strip_punc(split_token)
        split_token = split_token.replace("|", "-")
        split_token = re.sub(r'-+', "-", split_token)
        split_tokens.append(split_token)
        token_start = token_end + 1

    return split_tokens


def filter_non_eval(split_tokens, eval_tokens):
    tokens = [split_token.replace("-", "") for split_token in split_tokens]
    eval_splits = []

    j = 0
    for i, token in enumerate(tokens):
        if token == eval_tokens[j]:
            eval_splits.append(split_tokens[i])
            j += 1
            if j == len(eval_tokens):
                break
        else:
            pass

    return eval_splits


def save_segments(output_path, eval_tokens, segments):
    output_file = open(output_path, "w")
    output_file.close()
    output_file = open(output_path, "a")

    print("tokens evaluated", eval_tokens)
    print("model splits evaluated", segments)
    print(len(eval_tokens), len(segments))

    assert len(eval_tokens) == len(segments)
    for i, token in enumerate(eval_tokens):
        output_file.write(token + "\t" + segments[i] + "\n")
    output_file.close()


def load_targets(target_path):
    targets = []
    with open(target_path) as file:
        for line in file:
            target_splits = line.lower().split()[2].split("-")
            targets.append(target_splits)
    return targets


def load_predicts(predict_path):
    predicts = []
    with open(predict_path) as file:
        for line in file:
            predict_splits = line.lower().split()[1].split("-")
            predicts.append(predict_splits)
    return predicts


def eval_segments(predicts, targets, mode="boundaries"):

    if mode == "boundaries":
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for pred, targ in zip(predicts, targets):
            pred_bounds = []
            cur_index = -1
            for p in pred[0: -1]:
                pred_bounds.append(cur_index + len(p))
                cur_index += len(p)
            pred_non_bounds = [index for index in range(len("".join(pred)) - 1) if index not in pred_bounds]

            targ_bounds = []
            cur_index = -1
            for t in targ[0: -1]:
                targ_bounds.append(cur_index + len(t))
                cur_index += len(t)
            targ_non_bounds = [index for index in range(len("".join(targ)) - 1) if index not in targ_bounds]

            tp += len(set(pred_bounds) & set(targ_bounds))
            fp += len(set(pred_bounds) & set(targ_non_bounds))
            tn += len(set(pred_non_bounds) & set(targ_non_bounds))
            fn += len(set(pred_non_bounds) & set(targ_bounds))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 / (1 / precision + 1 / recall)

    elif mode == "morphemes":
        correct = 0.0
        for pred, targ in zip(predicts, targets):
            for p in pred:
                if p in targ:
                    correct += 1

        predicted_length = sum([len(pred) for pred in predicts])
        target_length = sum([len(targ) for targ in targets])
        precision, recall = correct / predicted_length, correct / target_length
        f_score = 2 / (1 / precision + 1 / recall)

    return (precision, recall, f_score)


def evaluate(model, char_vocab, eval_corpus, bptt_len, eval_data_path, output_path, device, segment_mode):

    text, split_indices = get_split_indices(model, char_vocab, eval_corpus, bptt_len, device, segment_mode)
    eval_tokens = get_eval_tokens(eval_data_path)

    tmp_eval_tokens = eval_tokens
    eval_tokens = []
    for token in tmp_eval_tokens:
        tmp_token = ""
        for ch in token:
            if ch in char_vocab.itos:
                tmp_token += ch
        eval_tokens.append(tmp_token)

    eval_tokens = [token[0:-1] if token[-1] == "." else token for token in eval_tokens]  # remove punctuation
    split_tokens = split(text, split_indices)

    print("token splits that are evaluated", split_tokens)
    eval_splits = filter_non_eval(split_tokens, eval_tokens)
    save_segments(output_path, eval_tokens, eval_splits)

    tmp_targets = load_targets(eval_data_path)
    targets = []
    for word in tmp_targets:
        targets.append([])
        for t in word:
            targets[-1].append(''.join([ch for ch in t if ch.isalpha() or ch.isnumeric() or ch=="%"]))

    predicts = load_predicts(output_path)

    for pred, targ in zip(predicts, targets):
        assert "".join(pred) == "".join(targ)

    predicts_unrolled = [p for pred in predicts for p in pred]
    targets_unrolled = [t for targ in targets for t in targ]
    print(predicts_unrolled)
    print(targets_unrolled)

    predicts_avg = sum(map(len, predicts_unrolled)) / len(predicts_unrolled)
    targets_avg = sum(map(len, targets_unrolled)) / len(targets_unrolled)

    print("predicts", predicts_avg)
    print("targets", targets_avg)

    morphemes_result = eval_segments(predicts, targets, mode="morphemes")
    boundaries_result = eval_segments(predicts, targets, mode="boundaries")
    result = morphemes_result + boundaries_result

    result = [100.0 * num for num in list(result)]
    result = ["%.2f" % num for num in result]
    print(result)


if __name__ == "__main__":
    main()