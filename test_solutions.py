import numpy as np
import pytest

from grader_contracts.python_basics import (
    TextInput, PositiveIntegerInput, VectorPairInput,
)
from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput, BinarizeInput, MatrixInput,
    RandomMatrixInput, ChessInput, RectangleInput,
    EllipseInput, TimeSeriesInput, OneHotInput,
)

from python_basics import (
    count_vowels, has_unique_characters, count_one_bits,
    multiplicative_persistence, mse, prime_factorization,
    pyramid, is_balanced_number,
)

from numpy_tasks import (
    sum_prod, binarize, unique_rows, unique_columns,
    matrix_statistics, chess, draw_rectangle, draw_ellipse,
    analyze_time_series, one_hot,
)

@pytest.mark.parametrize("text, expected", [
    ("Hello World", 3),
    ("AEIOU", 5),
    ("bcdfg", 0),
    ("", 0),
])
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


@pytest.mark.parametrize("text, expected", [
    ("abc", True),
    ("hello", False),
    ("", True),
    ("a", True),
    ("Aa", True),
])
def test_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


@pytest.mark.parametrize("number, expected", [
    (1, 1),
    (7, 3),
    (8, 1),
    (15, 4),
    (1024, 1),
])
def test_count_one_bits(number, expected):
    assert count_one_bits(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize("number, expected", [
    (39, 3),
    (4, 0),
    (999, 4),
    (10, 1),
    (25, 2),
])
def test_multiplicative_persistence(number, expected):
    assert multiplicative_persistence(
        PositiveIntegerInput(number)
    ) == expected


@pytest.mark.parametrize("predicted, expected, result", [
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([1, 2, 3], [2, 3, 4], 1.0),
    ([0], [5], 25.0),
    ([-1, -2], [1, 2], 10.0),
])
def test_mse(predicted, expected, result):
    data = VectorPairInput(predicted, expected)
    assert mse(data) == pytest.approx(result)


@pytest.mark.parametrize("number, expected", [
    (86240, "(2**5)(5)(7**2)(11)"),
    (2, "(2)"),
    (7, "(7)"),
    (12, "(2**2)(3)"),
    (100, "(2**2)(5**2)"),
    (1, ""),
])
def test_prime_factorization(number, expected):
    assert prime_factorization(
        PositiveIntegerInput(number)
    ) == expected


@pytest.mark.parametrize("number, expected", [
    (1, 1),
    (5, 2),
    (14, 3),
    (30, 4),
    (2, "It is impossible"),
    (10, "It is impossible"),
])
def test_pyramid(number, expected):
    assert pyramid(PositiveIntegerInput(number)) == expected


@pytest.mark.parametrize("number, expected", [
    (1234006, True),
    (123456, False),
    (121, True),
    (123, False),
    (11, True),
    (1234, False),
    (7, True),
])
def test_balanced_number(number, expected):
    assert is_balanced_number(
        PositiveIntegerInput(number)
    ) is expected




def test_sum_prod():
    matrices = np.array([
        [[1, 0], [0, 1]],
        [[2, 0], [0, 2]],
    ])
    vectors = np.array([
        [[1], [2]],
        [[3], [4]],
    ])

    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, [[7], [10]])
    assert result.shape == (2, 1)


def test_sum_prod_single_matrix():
    matrices = np.array([[[3]]])
    vectors = np.array([[[4]]])

    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    np.testing.assert_array_equal(result, [[12]])


def test_binarize():
    matrix = np.array([[0.2, 0.5], [0.7, 1.0]])
    original = matrix.copy()

    result = binarize(BinarizeInput(matrix, 0.5))

    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(matrix, original)


def test_binarize_negative_values():
    matrix = np.array([[-2, -1], [0, 2]])
    result = binarize(BinarizeInput(matrix, -1))

    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])


def test_unique_rows():
    matrix = np.array([[1, 2, 1], [3, 3, 4]])
    result = unique_rows(MatrixInput(matrix))

    assert result == [[1, 2], [3, 4]]


def test_unique_rows_single_element():
    matrix = np.array([[5]])
    assert unique_rows(MatrixInput(matrix)) == [[5]]


def test_unique_columns():
    matrix = np.array([[1, 2], [1, 3], [4, 2]])
    result = unique_columns(MatrixInput(matrix))

    assert result == [[1, 4], [2, 3]]


def test_unique_columns_single_element():
    matrix = np.array([[7]])
    assert unique_columns(MatrixInput(matrix)) == [[7]]


def test_matrix_statistics():
    data = RandomMatrixInput(3, 4, seed=42)
    result = matrix_statistics(data)

    assert result.matrix.shape == (3, 4)
    assert result.row_means.shape == (3,)
    assert result.column_means.shape == (4,)
    assert result.row_variances.shape == (3,)
    assert result.column_variances.shape == (4,)

    np.testing.assert_allclose(
        result.row_means, result.matrix.mean(axis=1)
    )
    np.testing.assert_allclose(
        result.column_variances, result.matrix.var(axis=0)
    )


def test_matrix_statistics_seed():
    data = RandomMatrixInput(2, 3, seed=123)

    first = matrix_statistics(data)
    second = matrix_statistics(data)

    np.testing.assert_array_equal(first.matrix, second.matrix)


def test_chess():
    data = ChessInput(3, 4, 0, 1)
    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])

    np.testing.assert_array_equal(chess(data), expected)


def test_chess_single_cell():
    result = chess(ChessInput(1, 1, 5, 9))
    np.testing.assert_array_equal(result, [[5]])


def test_draw_rectangle():
    data = RectangleInput(
        width=2,
        height=2,
        image_height=6,
        image_width=6,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )

    image = draw_rectangle(data)

    assert image.shape == (6, 6, 3)
    assert np.all(image[2:4, 2:4] == [255, 0, 0])
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0])


def test_draw_rectangle_full_image():
    data = RectangleInput(
        3, 3, 3, 3,
        (255, 255, 255),
        (0, 0, 0),
    )

    image = draw_rectangle(data)
    assert np.all(image == 255)


def test_draw_ellipse():
    data = EllipseInput(
        semi_axis_x=3,
        semi_axis_y=2,
        image_height=9,
        image_width=9,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )

    image = draw_ellipse(data)

    assert image.shape == (9, 9, 3)
    np.testing.assert_array_equal(image[4, 4], [255, 0, 0])
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0])


def test_draw_ellipse_symmetry():
    data = EllipseInput(
        2, 2, 7, 7,
        (255, 255, 255),
        (0, 0, 0),
    )

    image = draw_ellipse(data)

    np.testing.assert_array_equal(image, image[::-1, :, :])
    np.testing.assert_array_equal(image, image[:, ::-1, :])


def test_time_series():
    values = np.array([1, 3, 2, 5, 1])
    result = analyze_time_series(TimeSeriesInput(values, 3))

    assert result.mean == pytest.approx(2.4)
    assert result.variance == pytest.approx(np.var(values))
    assert result.std == pytest.approx(np.std(values))

    np.testing.assert_array_equal(
        result.local_maxima_indices, [1, 3]
    )
    np.testing.assert_array_equal(
        result.local_minima_indices, [2]
    )
    np.testing.assert_allclose(
        result.moving_average,
        [2.0, 10 / 3, 8 / 3],
    )


def test_time_series_window_one():
    values = np.array([2, 4, 6])
    result = analyze_time_series(TimeSeriesInput(values, 1))

    np.testing.assert_allclose(result.moving_average, values)
    assert len(result.local_maxima_indices) == 0
    assert len(result.local_minima_indices) == 0


def test_one_hot():
    data = OneHotInput([0, 2, 3, 0])
    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])

    np.testing.assert_array_equal(one_hot(data), expected)


def test_one_hot_class_count():
    data = OneHotInput([0, 1], class_count=4)
    result = one_hot(data)

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(
        result, [[1, 0, 0, 0], [0, 1, 0, 0]]
    )


def test_one_hot_empty():
    result = one_hot(OneHotInput([], class_count=3))
    assert result.shape == (0, 3)
