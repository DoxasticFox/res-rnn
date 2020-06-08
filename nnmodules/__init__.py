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

class Shift(torch.nn.Module):
    def __init__(self):
        super(Shift, self).__init__()

    def forward(self, x, s):
        x_original_width = x.size(-1)

        x, s = _broadcast_but_last(x, s)
        x = torch.cat((s, x), dim=-1)
        x = x[..., :x_original_width]
        return x

class Res(torch.nn.Module):
    def __init__(self, width, identity_proportion):
        super(Res, self).__init__()

        self.identity_proportion = identity_proportion

        self.fc1  = torch.nn.Linear(width, width)
        self.fc2  = torch.nn.Linear(width, width)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        r = x
        x = self.fc1(x).abs()
        x = \
            self.identity_proportion       * r + \
            (1 - self.identity_proportion) * self.fc2(x)
        return x

class ShiftedResNet(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_width,
            depth=50,
            identity_proportion=0.97
    ):
        super(ShiftedResNet, self).__init__()

        self.register_buffer('zero', torch.tensor([0.0]))

        self.fc1 = torch.nn.Linear(input_size, hidden_size)

        self.shf = torch.nn.ModuleList(
            Shift() for _ in range(depth))
        self.res = torch.nn.ModuleList(
            Res(hidden_size, identity_proportion) for _ in range(depth))

        self.fc2 = torch.nn.Linear(hidden_size, output_width)

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
    def __init__(self, input_size, state_size, output_size, identity_proportion=0.97):
        super(ResRnn, self).__init__()

        self.output_size = output_size
        self.stream_size = input_size + state_size
        self.identity_proportion = identity_proportion

        assert(output_size <= self.stream_size)

        self.register_buffer(
            'initial_output_stream',
            torch.zeros((self.stream_size,))
        )

        self.shf = Shift()
        self.res = Res(self.stream_size, self.identity_proportion)

    def forward(self, input):
        # input:         (seq_size, batch_size, input_size)
        # output_stream: (seq_size, batch_size, input_size + state_size)
        # returns:       (batch_size, output_size)

        output_stream = self.initial_output_stream

        for i in input:
            output_stream = self.shf(output_stream, i)
            output_stream = self.res(output_stream)

        # Truncate the output of the last RNN application. We take the last
        # output elements instead of the first because we hypothesise that this
        # will make learning long distance dependencies easier.
        output_stream = output_stream[..., -self.output_size:]

        return output_stream
