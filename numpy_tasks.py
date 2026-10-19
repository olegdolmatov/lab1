"""Задачи на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum(np.matmul(matrices, vectors), axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns = data.rows, data.columns
    mean, std, seed = data.mean, data.std, data.seed

    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))

    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = data.rows, data.columns
    first, second = data.first, data.second

    row_indices, col_indices = np.indices((rows, columns))
    return np.where((row_indices + col_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    x1 = (image_width - width) // 2
    y1 = (image_height - height) // 2

    x2 = x1 + width
    y2 = y1 + height

    image[max(0, y1):min(image_height, y2),
          max(0, x1):min(image_width, x2)] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    y, x = np.ogrid[:image_height, :image_width]

    center_x = (image_width - 1) / 2
    center_y = (image_height - 1) / 2

    mask = (
        ((x - center_x) / semi_axis_x) ** 2
        + ((y - center_y) / semi_axis_y) ** 2
        <= 1
    )

    image[mask] = shape_color
    return image



def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)

    mean = np.mean(values)
    variance = np.var(values)
    std = np.std(values)

    maxima = np.where(
        (values[1:-1] > values[:-2]) &
        (values[1:-1] > values[2:])
    )[0] + 1

    minima = np.where(
        (values[1:-1] < values[:-2]) &
        (values[1:-1] < values[2:])
    )[0] + 1

    moving_average = np.convolve(
        values, np.ones(window) / window, mode="valid"
    )

    return TimeSeriesStatistics(
        mean=float(mean),
        variance=float(variance),
        std=float(std),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )



def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)

    if class_count is None:
        class_count = int(np.max(labels)) + 1 if labels.size else 0

    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1

    return result
