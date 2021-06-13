import functools
import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from seaborn._core.data import PlotData


assert_series_equal = functools.partial(assert_series_equal, check_names=False)


class TestPlotData:

    @pytest.fixture
    def long_variables(self):
        variables = dict(x="x", y="y", hue="a", size="z", style="s_cat")
        return variables

    def test_long_df(self, long_df, long_variables):

        p = PlotData(long_df, long_variables)
        assert p._source_data is long_df
        assert p._source_vars is long_variables
        for key, val in long_variables.items():
            assert p.names[key] == val
            assert_series_equal(p.frame[key], long_df[val])

    def test_long_df_and_vectors(self, long_df, long_variables):

        long_variables["y"] = long_df["b"]
        long_variables["size"] = long_df["z"].to_numpy()

        p = PlotData(long_df, long_variables)

        assert_series_equal(p.frame["hue"], long_df[long_variables["hue"]])
        assert_series_equal(p.frame["y"], long_df["b"])
        assert_series_equal(p.frame["size"], long_df["z"])

        assert p.names["hue"] == long_variables["hue"]
        assert p.names["y"] == "b"
        assert p.names["size"] is None

    def test_long_df_with_index(self, long_df, long_variables):

        index = pd.Int64Index(np.arange(len(long_df)) * 2 + 10, name="i")
        long_variables["x"] = "i"
        p = PlotData(long_df.set_index(index), long_variables)

        assert p.names["x"] == "i"
        assert_series_equal(p.frame["x"], pd.Series(index, index))

    def test_long_df_with_multiindex(self, long_df, long_variables):

        index_i = pd.Int64Index(np.arange(len(long_df)) * 2 + 10, name="i")
        index_j = pd.Int64Index(np.arange(len(long_df)) * 3 + 5, name="j")
        index = pd.MultiIndex.from_arrays([index_i, index_j])
        long_variables.update({"x": "i", "y": "j"})

        p = PlotData(long_df.set_index(index), long_variables)
        assert_series_equal(p.frame["x"], pd.Series(index_i, index))
        assert_series_equal(p.frame["y"], pd.Series(index_j, index))

    def test_long_dict(self, long_dict, long_variables):

        p = PlotData(long_dict, long_variables)
        assert p._source_data is long_dict
        for key, val in long_variables.items():
            assert_series_equal(p.frame[key], pd.Series(long_dict[val]))

    @pytest.mark.parametrize(
        "vector_type",
        ["series", "numpy", "list"],
    )
    def test_long_vectors(self, long_df, long_variables, vector_type):

        variables = {key: long_df[val] for key, val in long_variables.items()}
        if vector_type == "numpy":
            variables = {key: val.to_numpy() for key, val in variables.items()}
        elif vector_type == "list":
            variables = {key: val.to_list() for key, val in variables.items()}

        p = PlotData(None, variables)

        assert list(p.names) == list(long_variables)
        if vector_type == "series":
            assert p._source_vars is variables
            assert p.names == {key: val.name for key, val in variables.items()}
        else:
            assert p.names == {key: None for key in variables}

        for key, val in long_variables.items():
            if vector_type == "series":
                assert_series_equal(p.frame[key], long_df[val])
            else:
                assert_array_equal(p.frame[key], long_df[val])

    def test_none_as_variable_value(self, long_df):

        p = PlotData(long_df, {"x": "z", "y": None})
        assert list(p.frame.columns) == ["x"]
        assert p.names == {"x": "z"}

    def test_long_undefined_variables(self, long_df):

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="not_in_df"))

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="x", y="not_in_df"))

        with pytest.raises(ValueError):
            PlotData(long_df, dict(x="x", y="y", hue="not_in_df"))

    def test_frame_and_vector_mismatched_lengths(self, long_df):

        vector = np.arange(len(long_df) * 2)
        with pytest.raises(ValueError):
            PlotData(long_df, {"x": "x", "y": vector})

    @pytest.mark.parametrize(
        "arg", [[], np.array([]), pd.DataFrame()],
    )
    def test_empty_data_input(self, arg):

        p = PlotData(arg, {})
        assert p.frame.empty
        assert not p.names

        if not isinstance(arg, pd.DataFrame):
            p = PlotData(None, dict(x=arg, y=arg))
            assert p.frame.empty
            assert not p.names

    def test_index_alignment_series_to_dataframe(self):

        x = [1, 2, 3]
        x_index = pd.Int64Index(x)

        y_values = [3, 4, 5]
        y_index = pd.Int64Index(y_values)
        y = pd.Series(y_values, y_index, name="y")

        data = pd.DataFrame(dict(x=x), index=x_index)

        p = PlotData(data, {"x": "x", "y": y})

        x_col_expected = pd.Series([1, 2, 3, np.nan, np.nan], np.arange(1, 6))
        y_col_expected = pd.Series([np.nan, np.nan, 3, 4, 5], np.arange(1, 6))
        assert_series_equal(p.frame["x"], x_col_expected)
        assert_series_equal(p.frame["y"], y_col_expected)

    def test_index_alignment_between_series(self):

        x_index = [1, 2, 3]
        x_values = [10, 20, 30]
        x = pd.Series(x_values, x_index, name="x")

        y_index = [3, 4, 5]
        y_values = [300, 400, 500]
        y = pd.Series(y_values, y_index, name="y")

        p = PlotData(None, {"x": x, "y": y})

        x_col_expected = pd.Series([10, 20, 30, np.nan, np.nan], np.arange(1, 6))
        y_col_expected = pd.Series([np.nan, np.nan, 300, 400, 500], np.arange(1, 6))
        assert_series_equal(p.frame["x"], x_col_expected)
        assert_series_equal(p.frame["y"], y_col_expected)

    def test_contains(self, long_df):

        p = PlotData(long_df, {"x": "y", "hue": long_df["a"]})
        assert "x" in p
        assert "y" not in p
        assert "hue" in p
