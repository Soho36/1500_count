import math


def suggested_size_per_account(K):
    exact = math.sqrt(K)
    return exact, math.floor(exact), math.ceil(exact)


for K in [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]:
    exact, f, c = suggested_size_per_account(K)
    print(f"K={K}: sqrt={exact:.3f} -> floor={f}, ceil={c}")
