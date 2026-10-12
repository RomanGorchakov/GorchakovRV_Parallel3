#!/usr/bin/env python3
from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot
import matplotlib.pyplot as plt
import time


start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()


def conjugate_gradient_method(A_part , b_part , x, N):
    r = empty(N, dtype=float64)
    q = empty(N, dtype=float64)

    s = 1
    p = zeros(N, dtype=float64)

    while s <= N:
        if s == 1:
            r_temp = dot(A_part.T, dot(A_part , x) - b_part)
            comm.Allreduce([r_temp , N, MPI.DOUBLE],
                           [r, N, MPI.DOUBLE], op=MPI.SUM)
        else:
            r -= q / dot(p, q)

        p += r / dot(r, r)
        
        q_temp = dot(A_part.T, dot(A_part , p))
        comm.Allreduce([q_temp , N, MPI.DOUBLE],
                       [q, N, MPI.DOUBLE], op=MPI.SUM)
        
        x -= p / dot(p, q)
        s += 1

        return x

def auxiliary_arrays_determination(M, numprocs):
    ave, res = divmod(M, numprocs - 1)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)

    rcounts[0] = 0
    displs[0] = 0

    for k in range(1, numprocs):
        if k <= res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave

        displs[k] = displs[k-1] + rcounts[k-1]

    return rcounts, displs
    
def main():
    if rank == 0:
        f1 = open('data/in.dat', 'r')
        N = array(int32(f1.readline()))
        M = array(int32(f1.readline()))
        f1.close()
    else:
        N = array(0, dtype=int32)
        
    comm.Bcast([N, 1, MPI.INT], root=0)
    
    if rank == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    else:
        rcounts_M, displs_M = None, None

    M_part = array(0, dtype=int32)
    comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)
    
    if rank == 0:
        f2 = open('data/AData.dat', 'r')
        for k in range(1, numprocs):
            A_part = empty((rcounts_M[k], N), dtype=float64)
            for j in range(rcounts_M[k]):
                for i in range(N):
                    A_part[j, i] = float64(f2.readline())

            comm.Send([A_part , rcounts_M[k]*N, MPI.DOUBLE], dest=k, tag=0)
        f2.close()
        A_part = empty((M_part , N), dtype=float64)
    else: # rank != 0
        A_part = empty((M_part , N), dtype=float64)
        comm.Recv([A_part , M_part*N, MPI.DOUBLE],
                  source=0, tag=0, status=None)
                  
    if rank == 0:
        b = empty(M, dtype=float64)
        f3 = open('data/bData.dat', 'r')
        for j in range(M):
            b[j] = float64(f3.readline())
        f3.close()
    else:
        b = None

    b_part = empty(M_part, dtype=float64)
    comm.Scatterv([b, rcounts_M , MPI.DOUBLE],
                  [b_part, M_part, MPI.DOUBLE], root=0)
    
    x = zeros(N, dtype=float64)
    x = conjugate_gradient_method(A_part, b_part, x, N)

    if rank == 0:
        plt.plot(arange(N), x, '-y', lw=3)
        plt.ylim=(-1.5, 1.5)
        plt.xlabel('i')
        plt.ylabel('x[i]')
        plt.show()
    
    elapsed = time.time() - start
    print(f"Время работы: {elapsed} с")

if __name__ == "__main__":
    main()