import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

sample_sizes = [10, 100, 1000]

def plot_distribution(dist_name, sizes, rvs_func, pdf_pmf_func, is_discrete=False, x_range=None, hist_range=None):
    """
    Вспомогательная функция для построения графиков для каждого распределения.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(dist_name, fontsize=16)

    for i, n in enumerate(sizes):
        ax = axes[i]
        
        sample = rvs_func(size=n)
        
        if is_discrete:
            x = np.arange(x_range[0], x_range[1] + 1)
            theoretical_vals = pdf_pmf_func(x)
            
            bins = np.arange(sample.min() - 0.5, sample.max() + 1.5, 1)
            ax.hist(sample, bins=bins, density=True, alpha=0.5, edgecolor='black', label='Эмпирическая')
            
            ax.plot(x, theoretical_vals, 'ro', markersize=4, label='Теоретическая (pmf)')
            ax.vlines(x, 0, theoretical_vals, colors='r', lw=1, alpha=0.5)
            ax.set_ylabel('Вероятность')
        else:
            x = np.linspace(x_range[0], x_range[1], 1000)
            theoretical_vals = pdf_pmf_func(x)
            
            if hist_range:
                ax.hist(sample, bins='auto', range=hist_range, density=True, alpha=0.5, edgecolor='black', label='Эмпирическая')
            else:
                ax.hist(sample, bins='auto', density=True, alpha=0.5, edgecolor='black', label='Эмпирическая')
                
            ax.plot(x, theoretical_vals, label='Теоретическая', color='C1')
            ax.set_ylabel('Плотность вероятности')
            
        if hist_range:
            ax.set_xlim(hist_range)
            
        ax.set_title(f'Размер выборки: n={n}')
        ax.set_xlabel('Значение')
        ax.legend()
        
    plt.tight_layout()
    plt.show()

plot_distribution(
    dist_name="Нормальное распределение N(0, 1)",
    sizes=sample_sizes,
    rvs_func=lambda size: stats.norm.rvs(loc=0, scale=1, size=size),
    pdf_pmf_func=lambda x: stats.norm.pdf(x, loc=0, scale=1),
    x_range=(-4, 4)
)

plot_distribution(
    dist_name="Распределение Коши C(0, 1)",
    sizes=sample_sizes,
    rvs_func=lambda size: stats.cauchy.rvs(loc=0, scale=1, size=size),
    pdf_pmf_func=lambda x: stats.cauchy.pdf(x, loc=0, scale=1),
    x_range=(-5, 5),
    hist_range=(-5, 5) 
)

scale_laplace = 1 / np.sqrt(2)
plot_distribution(
    dist_name="Распределение Лапласа L(0, 1/√2)",
    sizes=sample_sizes,
    rvs_func=lambda size: stats.laplace.rvs(loc=0, scale=scale_laplace, size=size),
    pdf_pmf_func=lambda x: stats.laplace.pdf(x, loc=0, scale=scale_laplace),
    x_range=(-5, 5)
)

mu_poisson = 10
plot_distribution(
    dist_name="Распределение Пуассона P(10)",
    sizes=sample_sizes,
    rvs_func=lambda size: stats.poisson.rvs(mu=mu_poisson, size=size),
    pdf_pmf_func=lambda x: stats.poisson.pmf(x, mu=mu_poisson),
    is_discrete=True,
    x_range=(0, 25)
)

loc_uniform = -np.sqrt(3)
scale_uniform = 2 * np.sqrt(3)
plot_distribution(
    dist_name="Равномерное распределение U(-√3, √3)",
    sizes=sample_sizes,
    rvs_func=lambda size: stats.uniform.rvs(loc=loc_uniform, scale=scale_uniform, size=size),
    pdf_pmf_func=lambda x: stats.uniform.pdf(x, loc=loc_uniform, scale=scale_uniform),
    x_range=(-3, 3)
)
