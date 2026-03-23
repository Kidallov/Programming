from functools import partial

# Период полураспада в годах
isotopes = {
    "C": 5730,
    "Cd": 9*10**15,
    "H": 12.3,
    "Np": 2.1 * 10**6
}

def remaining_amount(N, t_half, t):
    return N * ((1 / 2) ** (t / t_half))
N = 1000
t = 10000

for isotope, t_half in isotopes.items():

    calculate_remaining = partial(remaining_amount, N, t_half)
    remaining_N = calculate_remaining(t)
    print(f'От радиоактивного изотопа {isotope} после {t} лет осталось {remaining_N:.2f} единиц вещества.')
