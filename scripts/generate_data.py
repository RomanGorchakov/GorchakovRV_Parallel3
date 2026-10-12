#!/usr/bin/env python3
"""
Генерация тестовых данных для ЛР №2
"""
import numpy as np
import os

def generate_test_data(M, N, seed, output_dir):
    """Генерация тестовых данных"""
    np.random.seed(seed)
    
    A = np.random.rand(M, N)
    x = np.random.rand(N)
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.savetxt(os.path.join(output_dir, "AData.dat"), A.ravel()[:, None])
    np.savetxt(os.path.join(output_dir, "xData.dat"), x)
    
    with open(os.path.join(output_dir, "in.dat"), "w") as f:
        f.write(f"{N}\n{M}\n")

if __name__ == "__main__":
    generate_test_data(200, 100, 42, "data")