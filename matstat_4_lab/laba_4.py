import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Фиксируем seed для воспроизводимости
np.random.seed(42)

# Настройки для графиков
sample_sizes = [20, 60, 100]
x_cont = np.linspace(-4, 4, 1000)
x_disc = np.arange(6, 15)
x_disc_cont = np.linspace(6, 14, 1000)

# Словари с распределениями (названия и объекты scipy)
distributions = {
    'Нормальное N(0,1)': stats.norm(loc=0, scale=1),
    'Коши C(0,1)': stats.cauchy(loc=0, scale=1),
    'Лапласа L(0, 1/sqrt(2))': stats.laplace(loc=0, scale=1/np.sqrt(2)),
    'Пуассона P(10)': stats.poisson(mu=10), # Взято mu=10, чтобы пик был в [6, 14]
    'Равномерное U(-sqrt(3), sqrt(3))': stats.uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3))
}

# Функция для построения и сохранения графиков
def plot_distribution(name, dist, idx):
    is_discrete = 'Пуассона' in name
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Распределение: {name}', fontsize=16)
    
    for i, n in enumerate(sample_sizes):
        # Генерация выборки
        data = dist.rvs(size=n)
        data_sorted = np.sort(data)
        y_ecdf = np.arange(1, n + 1) / n
        
        # --- Верхний ряд: ЭФР ---
        ax_ecdf = axes[0, i]
        ax_ecdf.set_title(f'ЭФР (n={n})')
        
        if is_discrete:
            # Для дискретного теоретическая функция - тоже ступеньки
            ax_ecdf.step(data_sorted, y_ecdf, where='post', label=f'ЭФР (n={n})', color='blue')
            ax_ecdf.step(x_disc, dist.cdf(x_disc), where='post', label='Теор. ФР', color='red', linestyle='--')
            ax_ecdf.set_xlim(6, 14)
        else:
            ax_ecdf.step(data_sorted, y_ecdf, where='post', label=f'ЭФР (n={n})', color='blue')
            ax_ecdf.plot(x_cont, dist.cdf(x_cont), label='Теор. ФР', color='red', linestyle='--')
            ax_ecdf.set_xlim(-4, 4)
            
        ax_ecdf.set_ylim(-0.05, 1.05)
        ax_ecdf.grid(True, alpha=0.3)
        ax_ecdf.legend(fontsize=8)
        
        # --- Нижний ряд: Плотности и гистограммы ---
        ax_dens = axes[1, i]
        ax_dens.set_title(f'Оценка плотности (n={n})')
        
        # Ядерная оценка плотности (KDE)
        try:
            kde = stats.gaussian_kde(data)
            x_vals = x_disc_cont if is_discrete else x_cont
            ax_dens.plot(x_vals, kde(x_vals), label='Ядерная оценка', color='blue')
        except np.linalg.LinAlgError:
            pass # Если вдруг сингулярная матрица
            
        # Гистограмма
        if is_discrete:
            bins = np.arange(5.5, 15.5, 1)
            ax_dens.hist(data, bins=bins, density=True, alpha=0.3, color='gray', label='Гистограмма')
            ax_dens.plot(x_disc, dist.pmf(x_disc), 'ro', label='Теор. плотность (вероятности)')
            ax_dens.vlines(x_disc, 0, dist.pmf(x_disc), colors='red', linestyles='-', alpha=0.5)
            ax_dens.set_xlim(6, 14)
        else:
            ax_dens.hist(data, bins='auto', density=True, alpha=0.3, color='gray', label='Гистограмма')
            ax_dens.plot(x_cont, dist.pdf(x_cont), label='Теор. плотность', color='red', linestyle='--')
            ax_dens.set_xlim(-4, 4)
            
        ax_dens.grid(True, alpha=0.3)
        ax_dens.legend(fontsize=8)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    filename = f'fig{idx}_{name.split()[0].lower()}.png'
    plt.savefig(filename, dpi=200)
    plt.close()

# Запуск генерации
for idx, (name, dist) in enumerate(distributions.items(), 1):
    plot_distribution(name, dist, idx)
    print(f"Сохранен график для: {name}")