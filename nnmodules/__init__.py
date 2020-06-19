import collections
import itertools
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
        x = torch.cat((s, x), dim=-1)

        return x

class Res(torch.nn.Module):
    def __init__(self, width, linearity):
        super(Res, self).__init__()

        self.linearity = linearity

        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
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

class ShiftedResNet(torch.nn.Module):
    def __init__(
            self,
            input_width,
            hidden_width,
            output_width,
            depth=100,
            linearity=0.97
    ):
        super(ShiftedResNet, self).__init__()

        self.register_buffer('zero', torch.tensor([0.0]))

        self.fc1 = torch.nn.Linear(input_width, hidden_width)

        self.shf = torch.nn.ModuleList(
            ShiftRight() for _ in range(depth))
        self.res = torch.nn.ModuleList(
            Res(hidden_width, linearity) for _ in range(depth))

        self.fc2 = torch.nn.Linear(hidden_width, output_width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)

        for s, r in zip(self.shf, self.res):
            x = s(x, self.zero.expand((25,)))
            x = r(x)

        x = self.fc2(x)

        return x

class ResRnn(torch.nn.Module):
    def __init__(
        self,
        input_width,
        state_width,
        output_width,
        linearity=0.99999
    ):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.state_width = state_width
        self.output_width = output_width
        self.stream_width = input_width + state_width
        self.linearity = linearity

        assert(self.output_width <= self.stream_width)
        assert(self.input_width >= 0)
        assert(self.state_width >= 0)

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
        elif seq_indices is None:
            pass
        else:
            raise ValueError(
                'seq_indices must be an int or list of ints whose length is the '
                'size of the batch'
            )

        if seq_indices is None:
            seq_index_to_batch_indices = None
        else:
            seq_indices = [
                seq_index if seq_index >= 0 else seq_width + seq_index
                for seq_index in seq_indices
            ]
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
        # output_state:
        #     (seq_width, batch_width, output_width)
        # returns:
        #     (seq_width, batch_width, output_width) and
        #     (seq_width, batch_width, state_width)

        # Validate inputs
        input_seq_width, input_batch_width, input_width = input.size()

        state_batch_width, state_width = None, None
        if state:
            try:
                state_batch_width, state_width = state.size()
            except ValueError:
                pass

            try:
                state_width = state.size()
            except ValueError:
                pass

        assert(                 input_width == self.input_width)
        assert(state == None or state_width == self.state_width)
        assert(state == None or state_batch_width == input_batch_width)
        assert(state == None or input_seq_width == state_batch_width)
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
        # easier.
        outputs = streams[..., -self.output_width:]
        states  = streams[..., :-self.output_width]

        return outputs, states
