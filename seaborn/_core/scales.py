from __future__ import annotations

import pandas as pd
from matplotlib.scale import LinearScale

from .rules import categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from collections.abc import Sequence
    from matplotlib.scale import Scale
    from .typing import VariableType


class ScaleWrapper:

    def __init__(self, scale: Scale, type: VariableType):

        self._scale = scale
        transform = scale.get_transform()
        self.forward = transform.transform
        self.reverse = transform.inverted().transform
        self.type = type

    @property
    def order(self):
        if hasattr(self._scale, "order"):
            return self._scale.order
        return None

    def cast(self, data):
        if hasattr(self._scale, "cast"):
            return self._scale.cast(data)
        return data


class CategoricalScale(LinearScale):

    def __init__(self, axis: str, order: Optional[Sequence], formatter: Optional):
        # TODO what type is formatter?

        super().__init__(axis)
        self.order = order
        self.formatter = formatter

    def cast(self, data):

        data = pd.Series(data)
        order = pd.Index(categorical_order(data, self.order))
        if self.formatter is None:
            order = order.astype(str)
            data = data.astype(str)
        else:
            order = order.map(self.formatter)
            data = data.map(self.formatter)

        data = pd.Series(pd.Categorical(
            data, order, self.order is not None
        ), index=data.index)

        return data
