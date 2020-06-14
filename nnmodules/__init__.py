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
            Shift() for _ in range(depth))
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
        output_width=None,
        linearity=0.99999
    ):
        super(ResRnn, self).__init__()

        self.input_width = input_width
        self.state_width = state_width
        self.output_width = output_width
        self.stream_width = input_width + state_width
        self.linearity = linearity

        self.output_width = output_width or self.stream_width

        assert(self.output_width <= self.stream_width)
        assert(self.input_width >= 0)
        assert(self.state_width >= 0)
        assert(self.stream_width > 0)

        self.initial_stream = torch.nn.Parameter(
            torch.zeros(size=(self.stream_width,)),
        )
        self.do_terminate = None

        self.ins = Shift()
        self.res = Res(self.stream_width, self.linearity)

    def forward(
        self,
        input,
        output_indices=None,
        term_fun=None,
        max_iterations=None
    ):
        # input:
        #     (seq_width, batch_width, input_width) or
        #     (batch_width, state_width)
        # output_stream:
        #     (seq_width, batch_width, output_width)
        # returns:
        #      if output_indices == -1:
        #          (batch_width, output_width)
        #      elif output_indices == None:
        #          (seq_length, batch_width, output_width)
        #      elif output_indices is list-like:
        #          (batch_width, output_width)

        # Check inputs and unpack input size
        seq_width, batch_width, input_width, state_width = 0, 0, 0, 0
        try:
            seq_width, batch_width, input_width = input.size()
        except ValueError:
            pass
        try:
            batch_width, state_width = input.size()
        except ValueError:
            pass

        assert(input_width == 0 or input_width == self.input_width)
        assert(
            output_indices is None or
            output_indices == -1 or
            output_indices.size() == torch.Size((batch_width,))
        )
        assert(
            type(output_indices) != torch.Tensor or
            output_indices.min() >= 0)
        assert(
            type(output_indices) != torch.Tensor or
            output_indices.max() < seq_width)
        assert(
            input_width > 0 or
            term_fun is not None or
            max_iterations is not None
        )

        # Apply RNN
        if input_width == 0 and self.do_terminate is None and term_fun:
            self.do_terminate = torch.tensor(
                [False] * batch_width,
                device=input.device
            )
        if input_width == 0 and term_fun:
            self.do_terminate = self.do_terminate.zero_()

        outputs = []
        for index in itertools.count():
            if input_width == 0 and index >= max_iterations:
                break
            if input_width == 0 and term_fun and outputs:
                self.do_terminate |= term_fun(outputs[-1])
                if self.do_terminate.all():
                    break
            if input_width > 0 and index == seq_width:
                break

            # Prepare first input to RNN
            if input_width == 0 and index == 0:
                output_stream = input
            if input_width > 0 and index == 0:
                output_stream = (1 - self.linearity) * self.initial_stream

            # Insert the input into the stream
            if input_width > 0:
                output_stream = self.ins(output_stream, input[index])

            # Perform a single ResRNN application
            output_stream = self.res(output_stream)

            # Take outputs from the high indices because it might help with
            # learning long distance dependencies.
            output = output_stream[..., -self.output_width:]
            outputs.append(output)

        # Collect the outputs we want.
        #   * Return the outputs from the last RNN application in every batch.
        #     If the sequences do not have the same length, the shorter
        #     sequences would have had the RNN applied to them a few extra times
        #     needlessly.
        if output_indices == -1:
            return outputs[-1]
        outputs = torch.stack(outputs)
        #   * Return the outputs from every RNN application in every batch.
        if output_indices is None:
            return outputs
        #   * Return the outputs from the last RNN application to each sequence
        #     in the batch.
        if type(output_indices) == torch.Tensor:
            return outputs[output_indices, torch.range(batch_width)]

        raise RuntimeError('Failed to collect output')
