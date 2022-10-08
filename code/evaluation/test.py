from utils import load_model, load_vocab
import argparse
import torch
from model import SubwordSegmentalLM
from train import map_chars2lex, tokenize_chars, evaluate
from data import Corpus
from torchtext.legacy.data import Field
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():

    max_seg_len = 10
    only_subword = True
    len_dist = "learned"
    hidden_size = 1024
    feedforward_size = 1024
    bptt_len = 120
    num_heads = 8
    num_layers = 3
    encoder_type = "lstm"
    lexicon = "true"
    separate_lines = False
    reg_exp = 0
    reg_coef = 0
    batch_size = 64
    lang = "nr"

    OUTPUT_DIR = "output/"
    LM_DATA_DIR = "/home/francois/uct/data/nguni_lm/nguni_lm/" + lang + "/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=OUTPUT_DIR + "sslm." + lang)
    parser.add_argument("--char_vocab_path", default=OUTPUT_DIR + "char_vocab" + "." + lang)
    parser.add_argument("--lex_vocab_path", default=OUTPUT_DIR + "lex_vocab" + "." + lang)
    parser.add_argument("--len_lex_vocab_path", default=OUTPUT_DIR + "len_lex_vocab" + "." + lang)
    parser.add_argument("--eval_corpus_path", default=LM_DATA_DIR + "test" + "." + lang)

    args = parser.parse_args()
    model_path = args.model_path
    char_vocab_path = args.char_vocab_path
    lex_vocab_path = args.lex_vocab_path
    len_lex_vocab_path = args.len_lex_vocab_path
    eval_corpus_path = args.eval_corpus_path
    char_vocab = load_vocab(char_vocab_path)
    char_vocab.unk_index = char_vocab.stoi[" "]

    if lexicon == "true":
        lex_vocab = load_vocab(lex_vocab_path)
        lex_vocab_size = len(lex_vocab.itos)

        lenlex = os.path.isfile(len_lex_vocab_path)
        if lenlex:
            lenlex_vocab = load_vocab(len_lex_vocab_path)
            lenlex_vocab_sizes = {}
            chars2lenlex = {}
            for seg_len in lenlex_vocab.lens:
                lenlex_vocab_sizes[seg_len] = len(lenlex_vocab.lens[seg_len].itos)
                chars2lenlex[seg_len] = map_chars2lex(char_vocab, lenlex_vocab.lens[seg_len])
                print(seg_len, lenlex_vocab.lens[seg_len].itos)
        else:
            lenlex_vocab_sizes = None
            chars2lenlex = None

        chars2lex = map_chars2lex(char_vocab, lex_vocab)
    else:
        lenlex = False
        lex_vocab_size = 0
        lenlex_vocab_sizes = None
        chars2lenlex = None
        chars2lex = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
              "dropout": 0.2, "max_seg_len": params_dict["max_seg_len"], "eom_id": char_vocab.stoi["<eom>"], "pad_id": char_vocab.stoi["<pad>"],
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
    val_loss, val_R, val_nll, lex_coefs = evaluate(model, char_vocab, eval_corpus, device, batch_size, bptt_len, reg_coef,
                                        separate_lines)

    lex_coefs = torch.cat(lex_coefs)
    data_dict = {"lexcoef": lex_coefs.tolist()}
    sns.set(font_scale=1.5)
    df = pd.DataFrame(data_dict)
    print(torch.mean(lex_coefs))
    sns.histplot(data=df, x="lexcoef")
    plt.show()
    print("| valid loss {:5.2f} | valid R {:5.2f} | valid nll {:5.2f} | valid ppl {:5.2f} | "
          "valid bpc {:5.3f} | ".format(val_loss, val_R, val_nll, math.exp(val_nll), val_nll / math.log(2)))


if __name__ == "__main__":
    main()