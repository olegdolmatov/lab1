"""Задачи на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    vowels = "aeiou"
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    total = sum(
        (p - e) ** 2
        for p, e in zip(predicted, expected)
    )

    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power > 0:
            if power == 1:
                result.append(f"({divisor})")
            else:
                result.append(f"({divisor}**{power})")

        divisor += 1

    if number > 1:
        result.append(f"({number})")

    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0

    while total < cube_count:
        k += 1
        total += k ** 2

    if total == cube_count:
        return k

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    length = len(digits)
    middle = length // 2

    if length % 2 == 0:
        left = digits[:middle - 1]
        right = digits[middle + 1:]
    else:
        left = digits[:middle]
        right = digits[middle + 1:]

    return sum(map(int, left)) == sum(map(int, right))
