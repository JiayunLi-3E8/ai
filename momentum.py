class Momentum:
    def __init__(self, value=0, r=0.3819660112501051) -> None:
        self._x = value
        self._v = 0
        self._r = r

    def __lshift__(self, other) -> None:
        self._v = self._r * self._v + other
        self._x -= self._v

    def __neg__(self):
        return -self._x

    def __add__(self, other):
        return self._x + other

    def __radd__(self, other):
        return other + self._x

    def __sub__(self, other):
        return self._x - other

    def __rsub__(self, other):
        return other - self._x

    def __mul__(self, other):
        return self._x * other

    def __rmul__(self, other):
        return other * self._x

    def __truediv__(self, other):
        return self._x / other

    def __rtruediv__(self, other):
        return other / self._x

    def __floordiv__(self, other):
        return self._x // other

    def __rfloordiv__(self, other):
        return other // self._x

    def __mod__(self, other):
        return self._x % other

    def __rmod__(self, other):
        return other % self._x
