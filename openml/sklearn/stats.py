class Distribution(object):
    def get_params(self):
        raise NotImplementedError()

    def __repr__(self):
        return '%s.%s(%s)' % (self.__module__,
                              self.__class__.__name__,
                              ', '.join(['%s=%s' % (p, v)
                                        for p, v in
                                         sorted(self.get_params().items())]))

class RandInt(Distribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def rvs(self, random_state):
        return random_state.randint(self.lower, self.upper)

    def get_params(self):
        return {'lower': self.lower, 'upper': self.upper}


class Uniform(Distribution):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def rvs(self, random_state):
        return random_state.uniform(self.lower, self.upper)

    def get_params(self):
        return {'lower': self.lower, 'upper': self.upper}


class LogUniform(Distribution):
    def __init__(self, base, exponent_lower, exponent_upper):
        self.base = base
        self.exponent_lower = exponent_lower
        self.exponent_upper = exponent_upper

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

    def rvs(self, random_state):
        number = random_state.randint(self.number) + 1
        return random_state.randint(self.lower, self.upper, number)

    def get_params(self):
        return {'number': self.number, 'lower': self.lower,
                'upper': self.upper}