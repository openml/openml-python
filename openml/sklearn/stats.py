class Distribution(object):
    def get_params(self):
        raise NotImplementedError()

    def __repr__(self):
        return '%s.%s(%s)' % (self.__module__,
                              self.__class__.__name__,
                              ', '.join(['%s=%s' % (p, v)
                                        for p, v in
                                         sorted(self.get_params().items())]))

class Unparametrized(Distribution):
    def get_params(self):
        return {}


class Discrete(Distribution):
    def __init__(self, values):
        self.values = values
        if self.values is None:
            self.values = []

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)

    def get_params(self):
        return {'values': self.values}


class RandInt(Distribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

        if self.lower is None or self.upper is None:
            self.lower = 0
            self.upper = 1

    def rvs(self, random_state):
        return random_state.randint(self.lower, self.upper + 1)

    def get_params(self):
        return {'lower': self.lower, 'upper': self.upper}


class Uniform(Distribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

        if self.lower is None or self.upper is None:
            self.lower = 0
            self.upper = 1

    def rvs(self, random_state):
        return random_state.uniform(self.lower, self.upper)

    def get_params(self):
        return {'lower': self.lower, 'upper': self.upper}


class LogUniform(Distribution):
    def __init__(self, base, exponent_lower, exponent_upper):
        self.base = base
        self.exponent_lower = exponent_lower
        self.exponent_upper = exponent_upper

        if self.base is None or self.exponent_lower is None or\
                        self.exponent_upper is None:
            self.base = 2
            self.exponent_lower = 0
            self.exponent_upper = 1

    def rvs(self, random_state):
        return self.base ** random_state.uniform(self.exponent_lower,
                                                 self.exponent_upper)

    def get_params(self):
        return {'base': self.base, 'exponent_lower': self.exponent_lower,
                'exponent_upper': self.exponent_upper}


class LogUniformInt(Distribution):
    def __init__(self, base, exponent_lower, exponent_upper):
        self.base = base
        self.exponent_lower = exponent_lower
        self.exponent_upper = exponent_upper

        if self.base is None or self.exponent_lower is None or \
                        self.exponent_upper is None:
            self.base = 2
            self.exponent_lower = 0
            self.exponent_upper = 1

    def rvs(self, random_state):
        return int(self.base ** random_state.uniform(self.exponent_lower,
                                                     self.exponent_upper) + 0.5)

    def get_params(self):
        return {'base': self.base, 'exponent_lower': self.exponent_lower,
                'exponent_upper': self.exponent_upper}


class MultipleUniformIntegers(Distribution):
    def __init__(self, number, lower, upper):
        self.number = number
        self.lower = lower
        self.upper = upper

        if self.number is None or self.lower is None or self.upper is None:
            self.number = 1
            self.lower = 0
            self.upper = 1

    def rvs(self, random_state):
        number = random_state.randint(self.number) + 1
        return random_state.randint(self.lower, self.upper, number)

    def get_params(self):
        return {'number': self.number, 'lower': self.lower,
                'upper': self.upper}