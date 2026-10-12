#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from mpi4py import MPI
import time


start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()


def auxiliary_arrays_determination(size, numprocs):
    """
    Разбиение size элементов между процессами 1..numprocs-1.
    Процесс 0 работает как координатор и получает 0 элементов.
    """
    if numprocs < 2:
        raise ValueError("Запустите программу минимум с двумя MPI-процессами.")

    ave, res = divmod(int(size), numprocs - 1)
    counts = np.zeros(numprocs, dtype=np.intc)
    displs = np.zeros(numprocs, dtype=np.intc)

    for proc in range(1, numprocs):
        counts[proc] = ave + (1 if proc <= res else 0)
        displs[proc] = displs[proc - 1] + counts[proc - 1]

    return counts, displs


def allreduce_scalar(value):
    """Суммирование скалярной величины по всем процессам через MPI.Allreduce."""
    send = np.array([value], dtype=np.float64)
    recv = np.zeros(1, dtype=np.float64)
    comm.Allreduce(send, recv, op=MPI.SUM)
    return recv[0]


def conjugate_gradient_method(A_part, b_part, x_part,
                              N, rcounts_N, displs_N,
                              tol=1e-10, max_iter=None):
    """
    Метод сопряжённых градиентов для задачи наименьших квадратов:
        min ||A*x - b||_2.
    Эквивалентная система нормальных уравнений:
        (A.T @ A) x = A.T @ b.

    A и b распределены по строкам между процессами.
    Части x, r и p распределены по элементам вектора.
    """
    if max_iter is None:
        max_iter = max(100, 10 * N)

    n_local = len(x_part)
    x_part = x_part.copy()

    # Локальный вклад в правую часть A.T @ b.
    rhs_local = A_part.T @ b_part
    rhs = np.zeros(N, dtype=np.float64)
    comm.Allreduce(rhs_local, rhs, op=MPI.SUM)

    # Начальное приближение x = 0 (его локальные части уже получены).
    # Собираем полный x на каждом процессе.
    x_full = np.zeros(N, dtype=np.float64)
    comm.Allgatherv(
        [x_part, n_local, MPI.DOUBLE],
        [x_full, rcounts_N, displs_N, MPI.DOUBLE]
    )

    # r = A.T b - A.T A x; при x=0 это A.T b.
    r_part = rhs[displs_N[rank]:displs_N[rank] + n_local].copy()
    p_part = r_part.copy()
    rr = allreduce_scalar(np.dot(r_part, r_part))
    rhs_norm = np.sqrt(allreduce_scalar(np.dot(
        rhs[displs_N[rank]:displs_N[rank] + n_local],
        rhs[displs_N[rank]:displs_N[rank] + n_local]
    )))
    threshold = tol * max(1.0, rhs_norm)

    iterations = 0
    for iteration in range(max_iter):
        if np.sqrt(max(rr, 0.0)) <= threshold:
            break

        # Каждый процесс получает полный вектор направления p.
        p_full = np.zeros(N, dtype=np.float64)
        comm.Allgatherv(
            [p_part, n_local, MPI.DOUBLE],
            [p_full, rcounts_N, displs_N, MPI.DOUBLE]
        )

        # q = A.T @ A @ p. Каждый процесс считает свой вклад по строкам A.
        Ap_part = A_part @ p_full
        q_local = A_part.T @ Ap_part
        q_full = np.zeros(N, dtype=np.float64)
        comm.Allreduce(q_local, q_full, op=MPI.SUM)
        q_part = q_full[displs_N[rank]:displs_N[rank] + n_local]

        p_dot_q = allreduce_scalar(np.dot(p_part, q_part))
        if not np.isfinite(p_dot_q) or p_dot_q <= 0.0:
            # Нулевое направление/вырожденная система: дальнейший шаг
            # не определён численно.
            break

        alpha = rr / p_dot_q
        x_part += alpha * p_part
        r_part -= alpha * q_part

        rr_new = allreduce_scalar(np.dot(r_part, r_part))
        iterations = iteration + 1
        if np.sqrt(max(rr_new, 0.0)) <= threshold:
            rr = rr_new
            break

        beta = rr_new / rr
        p_part = r_part + beta * p_part
        rr = rr_new

    # Финальный полный вектор нужен всем процессам для согласованного
    # завершения; результат затем отдельно собирается на процессе 0.
    comm.Allgatherv(
        [x_part, n_local, MPI.DOUBLE],
        [x_full, rcounts_N, displs_N, MPI.DOUBLE]
    )

    return x_part, iterations, np.sqrt(max(rr, 0.0))


def main():
    # 1. Процесс 0 читает N и M; размеры рассылаются всем процессам.
    if rank == 0:
        data_dir = Path("data")
        with (data_dir / "in.dat").open("r") as f:
            N = int(f.readline().strip())
            M = int(f.readline().strip())
        dims = np.array([N, M], dtype=np.intc)
    else:
        dims = np.zeros(2, dtype=np.intc)

    comm.Bcast(dims, root=0)
    N, M = map(int, dims)

    if N <= 0 or M <= 0:
        raise ValueError(f"Размеры должны быть положительными: N={N}, M={M}")

    # 2. Расчёт counts/displacements для строк A и элементов x.
    rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, numprocs)

    # Для Scatterv матрица A передаётся как плоский массив.
    sendcounts_A = (rcounts_M * N).astype(np.intc)
    displs_A = (displs_M * N).astype(np.intc)

    # 3. Процесс 0 читает A и b; Scatterv распределяет строки A и элементы b.
    if rank == 0:
        data_dir = Path("data")
        A = np.loadtxt(data_dir / "AData.dat", dtype=np.float64).reshape(M, N)
        b = np.loadtxt(data_dir / "bData.dat", dtype=np.float64).reshape(M)
        A_flat = np.ascontiguousarray(A.ravel())
    else:
        A = None
        b = None
        A_flat = None

    n_rows_local = int(rcounts_M[rank])
    A_part = np.empty((n_rows_local, N), dtype=np.float64)
    comm.Scatterv(
        [A_flat, sendcounts_A, displs_A, MPI.DOUBLE] if rank == 0 else None,
        A_part.ravel(),
        root=0
    )

    b_part = np.empty(n_rows_local, dtype=np.float64)
    comm.Scatterv(
        [b, rcounts_M, displs_M, MPI.DOUBLE] if rank == 0 else None,
        b_part,
        root=0
    )

    # 4. Начальное приближение x = 0 и распределение его частей.
    if rank == 0:
        x_initial = np.zeros(N, dtype=np.float64)
    else:
        x_initial = None

    n_x_local = int(rcounts_N[rank])
    x_part = np.empty(n_x_local, dtype=np.float64)
    comm.Scatterv(
        [x_initial, rcounts_N, displs_N, MPI.DOUBLE]
        if rank == 0 else None,
        x_part,
        root=0
    )

    # 5. Параллельный метод сопряжённых градиентов.
    x_part, iterations, residual_norm = conjugate_gradient_method(
        A_part, b_part, x_part, N, rcounts_N, displs_N
    )

    # 6. Сбор окончательного результата на процессе 0.
    if rank == 0:
        x_parallel = np.empty(N, dtype=np.float64)
    else:
        x_parallel = None

    comm.Gatherv(
        x_part,
        [x_parallel, rcounts_N, displs_N, MPI.DOUBLE] if rank == 0 else None,
        root=0
    )

    # 7. Проверка через numpy.linalg.lstsq на процессе 0.
    if rank == 0:
        x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
        difference = np.linalg.norm(x_parallel - x_numpy)
        relative_difference = difference / max(1.0, np.linalg.norm(x_numpy))
        residual_parallel = np.linalg.norm(A @ x_parallel - b)
        residual_numpy = np.linalg.norm(A @ x_numpy - b)

        np.savetxt("x_parallel.dat", x_parallel, fmt="%.16e")
        elapsed = time.time() - start

        print(f"Размер матрицы A: {M} x {N}")
        print(f"Число MPI-процессов: {numprocs}")
        print(f"Итераций метода сопряжённых градиентов: {iterations}")
        print(f"Норма невязки нормальных уравнений: {residual_norm}")
        print(f"||A*x_parallel - b||_2: {residual_parallel}")
        print(f"||A*x_numpy - b||_2:    {residual_numpy}")
        print(f"||x_parallel - x_numpy||_2: {difference}")
        print(f"Относительное различие решений: {relative_difference}")
        print(f"Время работы: {elapsed} с")
        print("Параллельное решение сохранено в x_parallel.dat")


if __name__ == "__main__":
    main()
