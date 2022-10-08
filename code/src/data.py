# coding=utf-8
"""
Purpose
    Provides an interface between models and data.
"""

from torchtext.legacy.datasets import LanguageModelingDataset
from torchtext.legacy.data import Iterator, Dataset, Batch
from torch.nn.utils.rnn import pad_sequence
import torch
import math
import itertools


class Corpus(LanguageModelingDataset):

    @classmethod
    def iters(cls, dataset, batch_size=2, bptt_len=20, device=0, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.
        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.
        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        return BPTTIterator(dataset, batch_size=batch_size, bptt_len=bptt_len, device=device)


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that shuffle and sort default to train and (not train).
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil((len(self.dataset[0].text) / self.batch_size - 1)
                         / self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device)
        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + 1 + seq_len]
                if i == 0:
                    init_ids = torch.full((1, self.batch_size), fill_value=TEXT.vocab.stoi[" "], device=self.device)
                    batch_target = torch.cat((batch_text[0].unsqueeze(0), batch_target), dim=0)
                    batch_text = torch.cat((init_ids, batch_text), dim=0)

                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return


class LineCorpus(LanguageModelingDataset):
    @classmethod
    def iters(cls, dataset, batch_size=2, bptt_len=20, device=0, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.
        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.
        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        return BPTTLineIterator(dataset, batch_size=batch_size, bptt_len=bptt_len, device=device)


class BPTTLineIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that shuffle and sort default to train and (not train).
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        self.num_batches = None
        super(BPTTLineIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        num_lines = len([item for item in self.dataset[0].text if item == "<eos>"])
        return math.ceil(num_lines / (self.batch_size))

    def __iter__(self):

        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        eos_token = "<eos>"
        pad_id = TEXT.vocab.stoi[TEXT.pad_token]
        sos_id = TEXT.vocab.stoi["<sos>"]
        eos_id = TEXT.vocab.stoi["<eos>"]

        text = [list(y) for x, y in itertools.groupby(text, lambda z: z == eos_token) if not x]

        numerical_data = []
        for line in text:
            numerical_line = [TEXT.vocab.stoi[s] for s in line]
            numerical_data.append(torch.tensor(numerical_line))
        data = pad_sequence(numerical_data, padding_value=pad_id)
        max_seq_len = data.shape[0]

        batches = list(torch.split(data, split_size_or_sections=self.batch_size, dim=1))
        self.num_batches = len(batches)

        batch_lens = []
        batch_max_lens = []
        batch_sizes = []
        for batch in batches:
            batch_sizes.append(batch.shape[1])
            lens = []
            for i in range(batch_sizes[-1]):
                pad_idx = ((batch[:, i] == pad_id).nonzero(as_tuple=True)[0])
                if len(pad_idx) > 0:
                    lens.append(pad_idx[0].item())
                else:
                    lens.append(max_seq_len)
            if len(lens) > 0:
                batch_max_lens.append(max(lens))
            else:
                batch_max_lens.append(max_seq_len)
            batch_lens.append(lens)

        while batches[-1].shape[1] < self.batch_size:
            batches[-1] = torch.cat((batches[-1], torch.full(size=(max_seq_len, 1), fill_value=pad_id)), dim=1)
        data = torch.cat(batches, dim=0)

        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            batch_num = 0
            for i in range(0, self.num_batches * max_seq_len, max_seq_len):
                self.iterations += 1
                seq_len = min(max_seq_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + seq_len]

                sos_ids = torch.full((1, self.batch_size), fill_value=sos_id)
                pad_ids = torch.full((1, self.batch_size), fill_value=pad_id)

                batch_target = torch.cat((batch_text[0].unsqueeze(0), batch_target), dim=0)
                batch_text = torch.cat((sos_ids, batch_text), dim=0)

                batch_text = batch_text[0: batch_max_lens[batch_num] + 1]
                batch_target = batch_target[0: batch_max_lens[batch_num]]

                batch_target = torch.cat((batch_target, pad_ids), dim=0)
                batch_target[batch_lens[batch_num], range(batch_sizes[batch_num])] = eos_id

                batch_num += 1

                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return

