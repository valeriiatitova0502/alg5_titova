import time  # Импорт модуля для измерения времени выполнения операций
import numpy as np  # Импорт библиотеки для работы с массивами (матрицами)
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков

def create_matrix(size):
    # Создает две матрицы A и B размером size x size и инициализирует их определенным образом.
    A = [[0] * size for _ in range(size)]
    B = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if j == i + 1:
                A[i][j] = 1
            elif i == j + 1:
                B[i][j] = (i + 1)

    return A, B

def matrix_multiply(A, B):
    # Выполняет умножение матриц A и B с использованием библиотеки NumPy.
    return np.dot(A, B)

def matrix_subtract(A, B):
    # Выполняет вычитание матриц B из матрицы A с использованием библиотеки NumPy.
    return np.subtract(A, B)

def matrix_inverse(A):
    # Находит обратную матрицу для матрицы A с использованием библиотеки NumPy.
    return np.linalg.inv(A)

def measure_time_manual(size):
    # Измеряет время выполнения определенных операций с матрицами для ручного метода.
    A, B = create_matrix(size)

    start_time_mul = time.time()
    AB = matrix_multiply(A, B)
    BA = matrix_multiply(B, A)
    elapsed_time_mul = time.time() - start_time_mul

    start_time_inv = time.time()
    result_inv = matrix_inverse(matrix_subtract(AB, BA))
    result_inv = [[round(val, 2) for val in row] for row in result_inv]
    elapsed_time_inv = time.time() - start_time_inv

    start_time_expr = time.time()
    _ = matrix_subtract(AB, BA)
    elapsed_time_expr = time.time() - start_time_expr

    start_time_full_expr = time.time()
    _ = matrix_inverse(_)
    elapsed_time_full_expr = time.time() - start_time_full_expr

    return elapsed_time_mul, elapsed_time_inv, elapsed_time_expr, elapsed_time_full_expr

def measure_time_builtin(size):
    # Измеряет время выполнения определенных операций с матрицами для встроенного метода.
    A, B = create_matrix(size)

    start_time_mul = time.time()
    AB = np.dot(A, B)
    BA = np.dot(B, A)
    elapsed_time_mul = time.time() - start_time_mul

    start_time_inv = time.time()
    result_inv = np.linalg.inv(AB - BA)
    result_inv = np.round(result_inv, 2)
    elapsed_time_inv = time.time() - start_time_inv

    start_time_expr = time.time()
    _ = AB - BA
    elapsed_time_expr = time.time() - start_time_expr

    start_time_full_expr = time.time()
    _ = np.linalg.inv(_)
    elapsed_time_full_expr = time.time() - start_time_full_expr

    return elapsed_time_mul, elapsed_time_inv, elapsed_time_expr, elapsed_time_full_expr

def plot_graphs(sizes):
    # Строит графики зависимости времени выполнения операций с матрицами от их размеров.
    time_multiply_manual = []
    time_inverse_manual = []
    time_expr_manual = []
    time_full_expr_manual = []

    time_multiply_builtin = []
    time_inverse_builtin = []
    time_expr_builtin = []
    time_full_expr_builtin = []

    for size in sizes:
        elapsed_time_mul_manual, elapsed_time_inv_manual, elapsed_time_expr_manual, elapsed_time_full_expr_manual = measure_time_manual(size)
        elapsed_time_mul_builtin, elapsed_time_inv_builtin, elapsed_time_expr_builtin, elapsed_time_full_expr_builtin = measure_time_builtin(size)

        time_multiply_manual.append(elapsed_time_mul_manual)
        time_inverse_manual.append(elapsed_time_inv_manual)
        time_expr_manual.append(elapsed_time_expr_manual)
        time_full_expr_manual.append(elapsed_time_full_expr_manual)

        time_multiply_builtin.append(elapsed_time_mul_builtin)
        time_inverse_builtin.append(elapsed_time_inv_builtin)
        time_expr_builtin.append(elapsed_time_expr_builtin)
        time_full_expr_builtin.append(elapsed_time_full_expr_builtin)

    # График 1: Время умножения матриц
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_multiply_manual, label='Ручной метод')
    plt.plot(sizes, time_multiply_builtin, label='Встроенный метод')
    plt.title('Зависимость времени умножения матриц от размера матрицы')
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время (сек)')
    plt.legend()
    plt.show()

    # График 2: Время получения обратной матрицы (AB-BA)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_inverse_manual, label='Ручной метод')
    plt.plot(sizes, time_inverse_builtin, label='Встроенный метод')
    plt.title('Зависимость времени получения обратной матрицы от размера матрицы (AB-BA)')
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время (сек)')
    plt.legend()
    plt.show()

    # График 3: Время вычисления матричного выражения без учета формирования матрицы
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_expr_manual, label='Ручной метод')
    plt.plot(sizes, time_expr_builtin, label='Встроенный метод')
    plt.title('Зависимость времени вычисления матричного выражения без учета формирования матрицы от размера матрицы')
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время (сек)')
    plt.legend()
    plt.show()

    # График 4: Время вычисления матричного выражения с учетом формирования матрицы
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_full_expr_manual, label='Ручной метод')
    plt.plot(sizes, time_full_expr_builtin, label='Встроенный метод')
    plt.title('Зависимость времени вычисления матричного выражения с учетом формирования матрицы от размера матрицы')
    plt.xlabel('Размер матрицы')
    plt.ylabel('Время (сек)')
    plt.legend()
    plt.show()

def main():
    sizes_console = [3, 5, 7]

    for size in sizes_console:
        A, B = create_matrix(size)

        print(f"\nМатрица A ({size}x{size}):")
        for row in A:
            print(row)

        print(f"\nМатрица B ({size}x{size}):")
        for row in B:
            print(row)

        # Ручной метод
        elapsed_time_mul, elapsed_time_inv, elapsed_time_expr, elapsed_time_full_expr = measure_time_manual(size)

        print(f"\nРезультат для ручного метода (AB) ({size}x{size}):")
        result_mul_manual = matrix_multiply(A, B)
        for row in result_mul_manual:
            print(row)

        print(f"\nРезультат для ручного метода (BA) ({size}x{size}):")
        result_mul_manual_ba = matrix_multiply(B, A)
        for row in result_mul_manual_ba:
            print(row)

        print(f"\nРезультат для ручного метода (AB - BA) ({size}x{size}):")
        result_subtract_manual = matrix_subtract(result_mul_manual, result_mul_manual_ba)
        for row in result_subtract_manual:
            print(row)

        print(f"\nРезультат для ручного метода (AB-BA)^(-1) ({size}x{size}):")
        result_manual = matrix_inverse(matrix_subtract(result_mul_manual, result_mul_manual_ba))
        for row in result_manual:
            print([round(val, 2) for val in row])

        # Встроенный метод
        elapsed_time_mul_builtin, elapsed_time_inv_builtin, elapsed_time_expr_builtin, elapsed_time_full_expr_builtin = measure_time_builtin(size)

        print(f"\nРезультат для встроенного метода (AB-BA)^(-1) ({size}x{size}):")
        for row in np.round(np.linalg.inv(np.dot(A, B) - np.dot(B, A)), 2):
            print(row)

    # Построение графиков
    sizes_for_graphs = list(range(100, 3001, 100))
    plot_graphs(sizes_for_graphs)

if __name__ == "__main__":
    main()