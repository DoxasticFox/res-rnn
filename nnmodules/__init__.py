import torch
import itertools

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

# TODO: Investigate doing this without cat; It uses lots of memory.
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

        # TODO: Investigate using fc1 in place of fc2. It led to a 10% reduction
        #       in memory usage for a network with state_size=512 and a batch
        #       size of 512. Removing fc2 entirely didn't seem to work.
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

    def forward(self, input, state=None):
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

        if state is None:
            stream = (1 - self.linearity) * self.initial_state
        else:
            stream = state

        streams = []
        for element in input:
            stream = self.ins(element, stream, self.stream_width)
            stream = self.res(stream)

            streams.append(stream)

        streams = torch.stack(streams)

        # We take the last output elements instead of the first because we
        # hypothesise that this will make learning long distance dependencies
        # easier.
        outputs = streams[..., -self.output_width:]
        states  = streams[..., :-self.output_width]

        return outputs, states
