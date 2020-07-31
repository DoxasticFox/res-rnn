import multiprocessing
import random
import torch

padding_byte = b'\x00'

def strings_2_tensor(strings, device=None):
    strings = [s.encode('utf-8') if type(s) == str else s for s in strings]
    longs = torch.LongTensor(strings)
    if device is not None:
        longs = longs.to(device)
    one_hots = torch.nn.functional.one_hot(longs, num_classes=256)
    one_hots = one_hots.float()
    return one_hots.permute(1, 0, 2)

def tensor_2_strings(t, null_terminate=True):
    t = t.permute(1, 0, 2)
    _, indices = t.max(2)
    indices = indices.tolist()
    bytearrays = map(bytearray, indices)
    bytearrays = (
        [ba.split(padding_byte)[0] for ba in bytearrays]
        if null_terminate
        else bytearrays
    )
    return [ba.decode('utf-8', errors='ignore') for ba in bytearrays]

def pad_bytes(s, target_len):
    if len(s) > target_len:
        raise ValueError('length of string exceeds target length')
    else:
        return s + padding_byte * (target_len - len(s))

def split_lines(lines):
    for line in lines:
        try:
            src, tgt = line.split('\t')
        except ValueError:
            continue
        yield Pair(src, tgt)

def strip_pairs(pairs):
    return (pair.map(str.strip) for pair in pairs)

def filter_blank_pairs(pairs):
    def go(pair):
        return pair.src and pair.tgt
    return filter(go, pairs)

def filter_periods(pairs):
    def go(pair):
        return '.' not in pair
    return filter(go, pairs)

def string_pairs_to_byte_pairs(pairs):
    return (pair.map(str.encode) for pair in pairs)

def filter_len(pairs, _min=None, _max=None):
    def go(pair):
        _len = len(pair)
        return \
                (_min is None or _len >= _min) and \
                (_max is None or _len <= _max)

    return filter(go, pairs)

def parse_lang_names(file_name):
    segments = file_name.split('.')
    if len(segments) < 2:
        return None

    lang_names = segments[-2].split('-')
    if len(lang_names) != 2:
        return None

    return lang_names

class Pair:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def map(self, fun):
        return Pair(fun(self.src), fun(self.tgt))

    def __iter__(self):
        yield self.src
        yield self.tgt

    def __len__(self):
        return len(self.src) + len(self.tgt)

    def __repr__(self):
        return 'Pair({}, {})'.format(self.src, self.tgt)

class PairLengthGroups:
    def __init__(self, pairs=None):
        self.pairs = []
        self.groups = {}

        if pairs is not None:
            self.extend(pairs)

    def append(self, pair):
        self.pairs.append(pair)

        _len = len(pair)
        if _len not in self.groups:
            self.groups[_len] = []
        self.groups[_len].append(pair)

    def extend(self, pairs):
        for pair in pairs:
            self.append(pair)

    def __repr__(self):
        return 'PairLengthGroups({})'.format(self.groups)

class Batch:
    def __init__(
            self,
            pairs,
            device=None,
    ):
        self.pairs = pairs
        self.device = device

        srcs, tgts = zip(*pairs)

        # Package batch
        src_lens = [len(s) + 1 for s in srcs]
        tgt_lens = [len(t) + 1 for t in tgts]

        max_src_len = max(src_lens)
        max_tgt_len = max(tgt_lens)

        self.srcs = self._pad_and_convert_to_tensor(srcs, max_src_len)
        self.tgts = self._pad_and_convert_to_tensor(tgts, max_tgt_len)

        self.src_lens = torch.tensor(src_lens, device=device)
        self.tgt_lens = torch.tensor(tgt_lens, device=device)

    def _pad_and_convert_to_tensor(self, seq, _len=None):
        if _len:
            seq = [pad_bytes(s, _len) for s in seq]
        return strings_2_tensor(seq, self.device)

    def map(self, fun):
        return Batch(
            [p.map(fun) for p in self.pairs],
            device=self.device
        )

class BatchGenerator:
    def __init__(
        self,
        batch_size,
        similar_lengths=False,
        corpus_file_name=None,
        min_line_len=2,
        max_line_len=1000,
        src_lang=None,
        tgt_lang=None,
        device=None,
    ):
        self.batch_size = batch_size
        self.similar_lengths = similar_lengths
        self.device = device
        corpus_file_name = corpus_file_name or 'data/europarl-v9.de-en.tsv'

        # Parse lang names
        parsed_lang_names = parse_lang_names(corpus_file_name)
        if parsed_lang_names is None and None in (src_lang, tgt_lang):
            raise ValueError(
                'src_lang and tgt_lang must be given as lang names could not '
                'be determined from the corpus_file_name'
            )
        self.src_lang, self.tgt_lang = parsed_lang_names

        # Read corpus
        with open(corpus_file_name) as corpus_file:
            corpus_data = corpus_file.readlines()

        # Pre-process corpus
        corpus_data = split_lines(corpus_data)
        corpus_data = strip_pairs(corpus_data)
        corpus_data = filter_blank_pairs(corpus_data)
        corpus_data = filter_periods(corpus_data)
        corpus_data = string_pairs_to_byte_pairs(corpus_data)
        corpus_data = filter_len(corpus_data, min_line_len, max_line_len)
        corpus_data = PairLengthGroups(corpus_data)

        self.corpus_data = corpus_data

    def _choose_random_len_group(self):
        rand_pair_index = random.randint(0, len(self.corpus_data.pairs) - 1)
        rand_group_index = len(self.corpus_data.pairs[rand_pair_index])
        return self.corpus_data.groups[rand_group_index]

    def _choose_random_batch_of_pairs(self):
        population = (
            self._choose_random_len_group()
            if self.similar_lengths
            else self.corpus_data.pairs
        )
        batch_size = min(self.batch_size, len(population))
        return random.choices(population=population, k=batch_size)

    def __iter__(self):
        while True:
            yield Batch(
                self._choose_random_batch_of_pairs(),
                device=self.device
            )
