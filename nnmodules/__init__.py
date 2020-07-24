import __main__
import collections
import itertools
import os
import torch

def _broadcast_but_last(x, y):
    if len(x.size()) == len(y.size()):
        return x, y

    if len(x.size()) > len(y.size()):
        bigger, smaller = x, y
    else:
        bigger, smaller = y, x

    new = smaller.view(
        (1,) * (len(bigger.size()) - len(smaller.size())) + \
        smaller.size()
    )
    new = new.expand(
        bigger.size()[:-len(smaller.size())] + \
        smaller.size()
    )

    if len(x.size()) > len(y.size()):
        return x, new
    else:
        return new, y


class ResAbs(torch.nn.Module):
    def __init__(self, out_features):
        super(ResAbs, self).__init__()

        self.bendiness = torch.nn.Parameter(torch.zeros(out_features))

        self.out_features = out_features

    def forward(self, x):
        return x + self.bendiness * x.abs()


class ResSin(torch.nn.Module):
    def __init__(self, out_features):
        super(ResSin, self).__init__()

        self.bendiness = torch.nn.Parameter(torch.zeros(out_features))

        tau = (torch.acos(torch.tensor(0.0)) * 4.0).item()
        w_min = - 2.0 * tau
        w_max =   2.0 * tau

        self.register_buffer(
            'inner_weight',
            ResSin._range_non_zero(out_features, w_min, w_max),
        )
        self.register_buffer(
            'outer_weight',
            ResSin._range_two(
                (out_features,),
                -1, -w_max, w_max, 1
            ) / self.inner_weight,
        )

        self.out_features = out_features

    def forward(self, x):
        bendy_bit = \
            self.bendiness * \
            self.outer_weight * \
            (x * self.inner_weight).sin()

        return x + bendy_bit

    @staticmethod
    def _range(num, lo, hi):
        assert(num >= 2)
        return torch.arange(num) / float(num - 1) * (hi - lo) + lo

    @staticmethod
    def _bisect(nums):
        if len(nums) == 1:
            l_num = nums[0] // 2
            r_num = nums[0] - l_num
            return l_num, r_num
        elif len(nums) == 2:
            return nums
        else:
            ValueError('Too many nums')

    @staticmethod
    def _range_two(nums, l_lo, l_hi, r_lo, r_hi):
        l_num, r_num = ResSin._bisect(nums)

        return torch.cat((
            ResSin._range(l_num, l_lo, l_hi),
            ResSin._range(r_num, r_lo, r_hi),
        ))

    @staticmethod
    def _range_non_zero(num, lo, hi):
        l_num, r_num = ResSin._bisect((num,))

        l_lo = lo
        l_hi = -1.0 / l_num
        r_lo =  1.0 / r_num
        r_hi = hi

        return ResSin._range_two((l_num, r_num), l_lo, l_hi, r_lo, r_hi)


class ResLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ResLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            ResLinear._initial_weight(in_features, out_features)
        )

        self.bias = torch.nn.Parameter(
            torch.zeros(out_features)
        )

    @staticmethod
    @torch.no_grad()
    def _initial_weight(in_features, out_features):
        permuted_out = torch.randperm(out_features)

        weight = torch.zeros(out_features, in_features)
        for i in range(min(out_features, in_features)):
            weight[i, i] = 1.0
        weight = weight[permuted_out, :]
        weight = weight * torch.sign(torch.randn_like(weight))

        return weight

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight.t())


class Add(torch.nn.Module):
    def __init__(self, width_large, width_small):
        super(Add, self).__init__()

        self.width_large = width_large
        self.width_small = width_small

        self.padding = None

    def forward(self, large, small):
        assert(large.size(-1) == self.width_large)
        assert(small.size(-1) == self.width_small)

        if self.padding is None:
            self.padding = torch.zeros(
                self.width_large - self.width_small,
                device=small.device
            )
        small, self.padding = _broadcast_but_last(small, self.padding)
        small = torch.cat((small, self.padding), dim=-1)
        return large + small


class Res(torch.nn.Module):
    def __init__(self, width):
        super(Res, self).__init__()

        self.fc1 = ResLinear(width, width)
        self.act = ResSin(width)
        self.fc2 = ResLinear(width, width)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ResRnn(torch.nn.Module):
    def __init__(
        self,
        input_width,
        stream_width,
        output_width,
        checkpoint_name='ResRnn',
    ):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.stream_width = stream_width
        self.output_width = output_width
        self.checkpoint_name = checkpoint_name

        assert(self.input_width >= 0)
        assert(self.stream_width >= 0)
        assert(self.output_width >= 0)

        # Variables for coordinating checkpointing
        self.checkpoint_dir = None
        self.num_checkpoints = 0

        self.initial_stream = torch.nn.Parameter(
            torch.zeros(self.stream_width),
        )

        self.ins = Add(self.stream_width, self.input_width)
        self.res = Res(self.stream_width)

    def _get_seq_to_batch_index_map(self, seq_indices, seq_width, batch_width):
        if type(seq_indices) is int:
            seq_indices = [seq_indices] * batch_width
        elif type(seq_indices) is list:
            assert(len(seq_indices) == batch_width)
            assert(all(type(seq_index) is int for seq_index in seq_indices))
        elif type(seq_indices) is torch.Tensor:
            assert(seq_indices.size() == (batch_width,))
            seq_indices = seq_indices.tolist()
        elif seq_indices is None:
            pass
        else:
            raise ValueError(
                'seq_indices must be an int, list of ints whose length is the '
                'size of the batch, or 1-D tensor whose length is the size of '
                'the batch'
            )

        if seq_indices is None:
            seq_index_to_batch_indices = None
        else:
            seq_indices = [
                seq_index if seq_index >= 0 else seq_width + seq_index
                for seq_index in seq_indices
            ]
            assert(all(0 <= seq_index < seq_width for seq_index in seq_indices))
            seq_index_to_batch_indices = collections.defaultdict(list)
            for batch_index, seq_index in enumerate(seq_indices):
                seq_index_to_batch_indices[seq_index].append(batch_index)

        return seq_index_to_batch_indices

    def forward(self, input, stream=None, seq_indices=-1):
        # input:
        #     (seq_width, batch_width, input_width) or
        # stream:
        #     (batch_width, stream_width) or
        #     (stream_width)
        # seq_indices:
        #     int or
        #     [int]
        # returns:
        #     (seq_width, batch_width, output_width) and
        #     (seq_width, batch_width, stream_width) or
        #     (           batch_width, output_width) and
        #     (           batch_width, stream_width) or

        # Validate inputs
        input_seq_width, input_batch_width, input_width = input.size()

        stream_batch_width, stream_width = None, None
        if stream is not None:
            try:
                (stream_width,) = stream.size()
            except ValueError:
                pass

            try:
                stream_batch_width, stream_width = stream.size()
            except ValueError:
                pass

        assert(input_width == self.input_width)
        assert(stream is None or stream_width == self.stream_width)
        assert(
            stream_batch_width is None or
            stream_batch_width == input_batch_width)
        assert(input_batch_width > 0)

        # Pre-process seq_indices
        seq_index_to_batch_indices = self._get_seq_to_batch_index_map(
            seq_indices,
            input_seq_width,
            input_batch_width
        )

        # Used to collect return values
        streams = (
            []
            if seq_index_to_batch_indices is None
            else [None] * input_batch_width
        )
        def append_to_streams(stream):
            if seq_index_to_batch_indices is None:
                streams.append(stream)
            else:
                for batch_index in seq_index_to_batch_indices[seq_index]:
                    streams[batch_index] = stream[batch_index]

        # Set initial stream
        if stream is None:
            stream = self.initial_stream

        # Apply RNN
        for seq_index, element in enumerate(input):
            stream = self.ins(stream, element)
            stream = self.res(stream)

            append_to_streams(stream)

        # Stack streams and ensure everything is going as planned
        streams = torch.stack(streams)
        assert(
            streams.size() in [
                (input_seq_width, input_batch_width, self.stream_width),
                (                 input_batch_width, self.stream_width),
            ]
        )

        # We take the last elements as the output instead of the first because
        # we hypothesise that this will make learning long distance dependencies
        # easier. We slice it in a verbose way to allow for zero-length slices.
        outputs = streams[
            ...,
            :self.output_width
        ]

        return outputs, streams

    def _set_checkpoint_dir_if_none(self, file_name):
        def candidate_paths(path_name):
            yield path_name
            yield from (path_name + '-' + str(i + 2) for i in itertools.count())
        def unique_path(path_name):
            for candidate_path in candidate_paths(path_name):
                if not os.path.exists(candidate_path):
                    return candidate_path

        # Set self.checkpoint_dir if it isn't already
        if self.checkpoint_dir is None:
            abs_save_dir = os.path.dirname(os.path.realpath(__main__.__file__))
            abs_file_name = (
                file_name
                if os.path.isabs(file_name)
                else os.path.join(abs_save_dir, file_name)
            )
            self.checkpoint_dir = unique_path(os.path.dirname(abs_file_name))

        checkpoint_file_name = os.path.join(
            self.checkpoint_dir,
            os.path.basename(file_name)
        )

    def save_checkpoint(self):
        file_name = 'checkpoints/{}/{}.pt'.format(
            self.checkpoint_name,
            self.num_checkpoints
        )

        self._set_checkpoint_dir_if_none(file_name)

        checkpoint_file_name = os.path.join(
            self.checkpoint_dir,
            os.path.basename(file_name)
        )

        print(
            'Saving checkpoint for {} to {}'.format(
                self.checkpoint_name,
                checkpoint_file_name
            )
        )

        # Create dir + checkpoint
        self.save(checkpoint_file_name)

        print('Checkpoint saved')

        self.num_checkpoints += 1

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        self.eval()
