import random
import torch

# TODO: .to(device)
# TODO: Add padding options. Choose a more specific class name for the
#       DataLoader

def bytes_2_float_lists(bytes):
    bit_strings = ['{0:0>8b}'.format(b) for b in bytes]
    return [bit_string_2_float_list(bs) for bs in bit_strings]

def bit_string_2_float_list(bit_string):
    return [float(b) for b in bit_string]

def float_lists_2_string(fl):
    rounded      = [[round(x) for x in xs] for xs in fl]
    strings      = [[str(x)   for x in xs] for xs in rounded]
    joined       = [''.join(xs)            for xs in strings]
    ints         = [int(x, 2)              for x  in joined]
    bytes        = [x.to_bytes(1, 'big')   for x  in ints]
    joined_bytes = b''.join(bytes)
    return joined_bytes.decode('utf-8')

def pad_string(s, target_len, padding_char=b'\x00'):
    if len(s) > target_len:
        raise ValueError('length of string exceeds target length')
    else:
        return s + padding_char * (target_len - len(s))

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

def string_pairs_to_byte_pairs(pairs):
    return (pair.map(str.encode) for pair in pairs)

def filter_len(pairs, min=None, max=None):
    def go(pair):
        _len = len(pair)
        return \
                (min is None or _len >= min) and \
                (max is None or _len <= max)

    return filter(go, pairs)

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

class DataLoader:
    def __init__(
        self,
        batch_size,
        corpus_file_name=None,
        min_line_len=2,
        max_line_len=1000,
    ):
        self.batch_size = batch_size
        corpus_file_name = corpus_file_name or 'data/europarl-v9.de-en.tsv'

        with open(corpus_file_name) as corpus_file:
            corpus_data = corpus_file.readlines()

        corpus_data = split_lines(corpus_data)
        corpus_data = strip_pairs(corpus_data)
        corpus_data = filter_blank_pairs(corpus_data)
        corpus_data = string_pairs_to_byte_pairs(corpus_data)
        corpus_data = filter_len(corpus_data, min_line_len, max_line_len)
        corpus_data = PairLengthGroups(corpus_data)

        self.corpus_data = corpus_data

    def _choose_random_len_group(self):
        rand_pair_index = random.randint(0, len(self.corpus_data.pairs) - 1)
        rand_group_index = len(self.corpus_data.pairs[rand_pair_index])
        return self.corpus_data.groups[rand_group_index]

    def _choose_random_batch_of_pairs(self):
        group = self._choose_random_len_group()
        batch_size = min(self.batch_size, len(group))
        return random.choices(population=group, k=batch_size)

    def __iter__(self):
        while True:
            srcs, tgts = zip(*self._choose_random_batch_of_pairs())

            src_lens = [len(s) for s in srcs]
            tgt_lens = [len(t) for t in tgts]

            max_src_len = max(src_lens)
            max_tgt_len = max(tgt_lens)

            srcs = (pad_string(s, max_src_len) for s in srcs)
            tgts = (pad_string(s, max_tgt_len) for s in tgts)

            srcs = [bytes_2_float_lists(s) for s in srcs]
            tgts = [bytes_2_float_lists(s) for s in tgts]

            yield \
                (torch.tensor(srcs).permute(1, 0, 2), torch.tensor(src_lens)), \
                (torch.tensor(tgts).permute(1, 0, 2), torch.tensor(tgt_lens))
