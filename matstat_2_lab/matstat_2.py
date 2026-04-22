import numpy as np
import scipy.stats as stats
import pandas as pd


def calculate_characteristics(data):
    """Вычисление 5 характеристик для одной выборки."""
    mean = np.mean(data)
    med = np.median(data)
    zR = (np.min(data) + np.max(data)) / 2
    zQ = (np.quantile(data, 0.25) + np.quantile(data, 0.75)) / 2
    ztr = stats.trim_mean(data, 0.1)
    return [mean, med, zR, zQ, ztr]


def run_simulation(dist_name, rvs_func, sizes=[10, 100, 1000], iters=1000):
    """Проведение 1000 экспериментов и расчет E(z) +- sqrt(D(z))."""
    char_names = ['Выборочное среднее', 'Медиана', 'zR', 'zQ', 'ztr']
    results = {name: [] for name in char_names}

    for n in sizes:

        iter_results = np.zeros((iters, 5))

        for i in range(iters):
            data = rvs_func(size=n)
            iter_results[i] = calculate_characteristics(data)

        E = np.mean(iter_results, axis=0)
        sqrt_D = np.std(iter_results, axis=0)

        # Форматируем строку как E +- sqrt(D) с округлением
        for j, name in enumerate(char_names):
            formatted_val = f"{E[j]:.2f} ± {sqrt_D[j]:.2f}"
            results[name].append(formatted_val)

    # Создаем красивую таблицу pandas для вывода
    df = pd.DataFrame(results, index=[f'n={n}' for n in sizes]).T
    df.columns.name = dist_name
    print(f"\n--- {dist_name} ---")
    print(df.to_string())
    return df


distributions = {
    'Нормальное N(0,1)': lambda size: stats.norm.rvs(loc=0, scale=1, size=size),
    'Коши C(0,1)': lambda size: stats.cauchy.rvs(loc=0, scale=1, size=size),
    'Лапласа L(0, 1/sqrt(2))': lambda size: stats.laplace.rvs(loc=0, scale=1 / np.sqrt(2), size=size),
    'Пуассона P(10)': lambda size: stats.poisson.rvs(mu=10, size=size),
    'Равномерное U(-sqrt(3), sqrt(3))': lambda size: stats.uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=size)
}

np.random.seed(42)  
for name, func in distributions.items():
    run_simulation(name, func)