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

class Res(torch.nn.Module):
    def __init__(self, width, linearity):
        super(Res, self).__init__()

        assert(width > 0)
        assert(0.0 <= linearity < 1.0)

        self.linearity = linearity

        self.randperm = torch.nn.Parameter(
            torch.randperm(width),
            requires_grad=False,
        )
        self.randsign = torch.nn.Parameter(
            torch.sign(
                torch.arange(width).remainder(2) - 0.5
            )[torch.randperm(width)],
            requires_grad=False,
        )
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x[:, self.randperm] * self.randsign

        r = x

        x = self.fc1(x)
        x = x.clamp(min=0)
        x = self.fc2(x)

        x = self.linearity * r + (1 - self.linearity) * x

        return x

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

class ResRnn(torch.nn.Module):
    def __init__(
        self,
        input_width,
        stream_width,
        output_width,
        linearity=0.99999,
        checkpoint_name='ResRnn',
    ):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.stream_width = stream_width
        self.output_width = output_width
        self.linearity = linearity
        self.checkpoint_name = checkpoint_name

        assert(self.input_width <= self.stream_width)
        assert(self.output_width <= self.stream_width)
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
        self.res = Res(self.stream_width, self.linearity)

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
            stream = (1 - self.linearity) * self.initial_stream

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
