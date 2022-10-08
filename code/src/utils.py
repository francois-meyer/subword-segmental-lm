import torch
import numpy as np
import pickle
import copy


class LexiconVocab:

    def __init__(self, full_vocab, max_seg_len, non_alpha):

        len_range = range(1, max_seg_len+1)
        len_vocabs = {seg_len: copy.deepcopy(full_vocab) for seg_len in len_range}
        specials = ['<unk>', '<pad>', '<eos>']

        # Collect segments for different lengths
        len_segs = {seg_len: [] for seg_len in range(1, max_seg_len + 1)}
        for seg in full_vocab.itos:
            if seg in specials:
                continue

            len_segs[len(seg)].append(seg)

        # Create separate vocabularies for different segment lengths
        for seg_len in len_vocabs:
            len_vocabs[seg_len].itos = specials + len_segs[seg_len]
            len_vocabs[seg_len].stoi = {seg: index for index, seg in enumerate(len_vocabs[seg_len].itos)}
            len_vocabs[seg_len].freqs = {seg: len_vocabs[seg_len].freqs[seg] for seg in len_vocabs[seg_len].itos}

        self.lens = len_vocabs


def save_model_and_vocab(model, vocab, model_path, vocab_path):
    torch.save(model.state_dict(), model_path)
    print("Model saved at %s" % model_path)

    with open(vocab_path, "wb") as vocab_file:
        pickle.dump(vocab, vocab_file, protocol=4)
    print("Vocab saved at %s" % vocab_path)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_vocab(vocab, vocab_path):
    with open(vocab_path, "wb") as vocab_file:
        pickle.dump(vocab, vocab_file, protocol=4)
    print("Vocab saved at %s" % vocab_path)


def load_model(model, model_path, map_location):
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    return model


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
    return vocab


def generate_text(model, model_type, vocab, start="I", gen_len=50, eos_token="<eos>"):

    with torch.no_grad():
        model.eval()
        chars = list(start)

        if model_type == "lstm":
            state_h, state_c = model.init_states(1)

        for i in range(gen_len):
            input_ids = torch.tensor([[vocab.stoi[ch]] for ch in chars])

            if model_type == "lstm":
                y_pred, (state_h, state_c) = model(input_ids, (state_h, state_c))
            else:  # model_type == "transformer"
                y_pred = model(input_ids)

            last_char_logits = y_pred[-1][0]
            p = torch.nn.functional.softmax(last_char_logits).detach().numpy()
            char_index = np.random.choice(len(last_char_logits), p=p)
            if vocab.itos[char_index] == eos_token:
                chars.append("\n")
            else:
                chars.append(vocab.itos[char_index])

        print("".join(chars))
