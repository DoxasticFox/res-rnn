import collections
import random
import torch
import typing

# TODO: Pad bytes, not strings
# TODO: Convert to torch tensors
# TODO: .to(device)

def bit_string_2_float_list(bit_string):
    return [float(b) for b in bit_string]

def string_2_float_lists(string):
    bytes = string.encode()
    bit_strings = ['{0:0>8b}'.format(b) for b in bytes]
    return [bit_string_2_float_list(bs) for bs in bit_strings]

def float_lists_2_string(fl):
    rounded      = [[round(x) for x in xs] for xs in fl]
    strings      = [[str(x)   for x in xs] for xs in rounded]
    joined       = [''.join(xs)            for xs in strings]
    ints         = [int(x, 2)              for x  in joined]
    bytes        = [x.to_bytes(1, 'big')   for x  in ints]
    joined_bytes = b''.join(bytes)
    return joined_bytes.decode('utf-8')

def pad_string(s, target_length, padding_char='\x00'):
    if len(s) > target_length:
        raise ValueError('length of string exceeds target length')
    else:
        return s + padding_char * (target_length - len(s))

def filter_len(lines, min=None, max=None):
    if min is not None: lines = filter(lambda line: len(line) >= min, lines)
    if max is not None: lines = filter(lambda line: len(line) <= max, lines)
    return lines

class LineLengthGroup:
    def __init__(self):
        self.len = None
        self.lines = []

    def append(self, line):
        if self.len is None:
            self.len = len(line)
            self.lines.append(line)
        elif self.len == len(line):
            self.lines.append(line)
        else:
            raise ValueError(
                'line has length {} but {} expected'
                .format(len(line), self.line)
            )

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        return 'LineLengthGroup({}, {})'.format(
            repr(self.len),
            repr(self.lines)
        )

class LineLengthGroups:
    def __init__(self, lines=None):
        self.groups = collections.defaultdict(LineLengthGroup)

        if lines is not None:
            self.extend(lines)

    def append(self, line):
        self.groups[len(line)].append(line)

    def extend(self, lines):
        for line in lines:
            self.append(line)

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, i):
        return self.groups[i]

class DataLoader:
    def __init__(
        self,
        batch_size,
        corpus_file_name=None,
        min_line_len=4,
        max_line_len=1000,
    ):
        self.batch_size = batch_size
        self.corpus_file_name = \
            corpus_file_name or \
            'data/europarl-v9.de-en.tsv'
        self.min_line_len = min_line_len
        self.max_line_len = max_line_len

        with open(self.corpus_file_name) as corpus_file:
            self.corpus_data = corpus_file.readlines()

        self.corpus_data = map(str.strip, self.corpus_data)

        self.corpus_data = filter_len(
            self.corpus_data,
            min_line_len,
            max_line_len
        )
        self.corpus_data = sorted(self.corpus_data, key=len)
        self.line_length_groups = LineLengthGroups(self.corpus_data)

    def _choose_random_length_group(self):
        group_lens = [
            group.len for group in self.line_length_groups.groups.values()
        ]
        return random.choices(
            list(self.line_length_groups.groups.values()),
            group_lens
        )[0]

    def _choose_random_lines(self):
        group = self._choose_random_length_group()
        batch_size = min(self.batch_size, len(group))
        return random.choices(population=group.lines, k=batch_size)

    def _choose_random_sentence_pairs(self):
        for line in self._choose_random_lines():
            yield line.split('\t')

    def __iter__(self):
        while True:
            srcs, tgts = zip(*self._choose_random_sentence_pairs())

            max_src_len = max(map(len, srcs))
            max_tgt_len = max(map(len, tgts))

            srcs = [pad_string(s, max_src_len) for s in srcs]
            tgts = [pad_string(s, max_tgt_len) for s in tgts]

            srcs = [string_2_float_lists(s) for s in srcs]
            tgts = [string_2_float_lists(s) for s in tgts]

            yield srcs, tgts
