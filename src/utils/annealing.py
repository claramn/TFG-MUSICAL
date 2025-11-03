# MIT License

# Copyright (c) 2023 Hubert Rybka

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

class Annealer:
    def __init__(self, total_steps, shape='linear', baseline=0.0, cyclical=False, disable=False):
        self.current_step = 0
        self.total_steps  = total_steps
        self.shape        = shape
        self.baseline     = baseline
        self.cyclical     = cyclical
        self.disable      = disable

    def __call__(self, kld):
        if self.disable:
            return kld
        return kld * self._slope()

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0

    def _slope(self):
        t = self.current_step / self.total_steps
        if   self.shape == 'linear':   y = t
        elif self.shape == 'cosine':   y = (math.cos(math.pi*(t-1)) + 1)/2
        elif self.shape == 'logistic': y = 1/(1+math.exp(self.total_steps/2 - self.current_step))
        else:                           y = 1.0
        return y*(1-self.baseline) + self.baseline
