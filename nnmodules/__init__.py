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

class ShiftRight(torch.nn.Module):
    def __init__(self):
        super(ShiftRight, self).__init__()

    def forward(self, s, x, output_width=None):
        output_width = output_width or x.size(-1)
        assert(output_width <= s.size(-1) + x.size(-1))
        assert(output_width >= s.size(-1))

        excess_width = s.size(-1) + x.size(-1) - output_width
        if excess_width > 0:
            x = x[..., :-excess_width]
        elif excess_width == 0:
            x = x
        else:
            raise RuntimeError('excess_width should not be negative')

        s, x = _broadcast_but_last(s, x)

        if s.size(-1) > 0:
            x = torch.cat((s, x), dim=-1)

        return x

class Res(torch.nn.Module):
    def __init__(self, width, linearity):
        super(Res, self).__init__()

        self.linearity = linearity

        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        r = x

        x = self.fc1(x)
        x = x.clamp(min=0)
        x = self.fc2(x)

        x = self.linearity * r + (1 - self.linearity) * x

        return x

class ResRnn(torch.nn.Module):
    def __init__(
        self,
        input_width,
        state_width,
        output_width,
        linearity=0.99999,
        checkpoint_name='ResRnn',
    ):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.state_width = state_width
        self.output_width = output_width
        self.stream_width = input_width + state_width
        self.linearity = linearity
        self.checkpoint_name = checkpoint_name

        assert(self.output_width <= self.stream_width)
        assert(self.input_width >= 0)
        assert(self.state_width >= 0)

        # Variables for coordinating checkpointing
        self.checkpoint_dir = None
        self.num_checkpoints = 0

        self.initial_state = torch.nn.Parameter(
            torch.zeros(size=(self.state_width,)),
        )

        self.ins = ShiftRight()
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

    def forward(self, input, state=None, seq_indices=-1):
        # input:
        #     (seq_width, batch_width, input_width) or
        # state:
        #     (batch_width, state_width) or
        #     (state_width)
        # seq_indices:
        #     int or
        #     [int]
        # returns:
        #     (seq_width, batch_width, output_width) and
        #     (seq_width, batch_width, state_width) or
        #     (           batch_width, output_width) and
        #     (           batch_width, state_width) or

        # Validate inputs
        input_seq_width, input_batch_width, input_width = input.size()

        state_batch_width, state_width = None, None
        if state is not None:
            try:
                state_width = state.size()
            except ValueError:
                pass

            try:
                state_batch_width, state_width = state.size()
            except ValueError:
                pass

        assert(input_width == self.input_width)
        assert(state is None or state_width == self.state_width)
        assert(
            state_batch_width is None or
            state_batch_width == input_batch_width)
        assert(input_batch_width > 0)

        # Pre-process seq_indices
        seq_index_to_batch_indices = self._get_seq_to_batch_index_map(
            seq_indices,
            input_seq_width,
            input_batch_width
        )

        # Set initial stream
        if state is None:
            stream = (1 - self.linearity) * self.initial_state
        else:
            stream = state

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

        # Apply RNN
        for seq_index, element in enumerate(input):
            stream = self.ins(element, stream, self.stream_width)
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
            self.stream_width - self.output_width:self.stream_width
        ]
        states  = streams[
            ...,
            :self.state_width
        ]

        return outputs, states

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
