import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Фиксируем seed для воспроизводимости
np.random.seed(42)

# Настройки распределений (генераторы)
distributions = {
    'Нормальное': stats.norm(loc=0, scale=1),
    'Коши': stats.cauchy(loc=0, scale=1),
    'Лапласа': stats.laplace(loc=0, scale=1/np.sqrt(2)),
    'Пуассона': stats.poisson(mu=5),
    'Равномерное': stats.uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3))
}

# Теоретические вероятности (из расчетов в отчете)
theoretical_probs = {
    'Нормальное': 0.007,
    'Коши': 0.156,
    'Лапласа': 0.0625,
    'Пуассона': 0.0137,
    'Равномерное': 0.000
}

sample_sizes = [20, 100]
num_iterations = 1000

# Функция для подсчета доли выбросов по методу Тьюки
def get_outliers_fraction(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers) / len(data)

# Сбор результатов
results = {}

# Подготовка графика
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
fig.suptitle('Боксплоты Тьюки для различных распределений', fontsize=16)

for idx, (name, dist) in enumerate(distributions.items()):
    results[name] = []
    plot_data = []
    
    for n in sample_sizes:
        fractions = []
        # Выборка для графика (одна из генераций)
        sample_for_plot = dist.rvs(size=n)
        plot_data.append(sample_for_plot)
        
        # Симуляция 1000 раз
        for _ in range(num_iterations):
            sample = dist.rvs(size=n)
            fractions.append(get_outliers_fraction(sample))
            
        mean_fraction = np.mean(fractions)
        results[name].append(mean_fraction)
        
    results[name].append(theoretical_probs[name])
    
    # Отрисовка боксплотов
    axes[idx].boxplot(plot_data, labels=[f'n={sample_sizes[0]}', f'n={sample_sizes[1]}'])
    axes[idx].set_title(name)
    axes[idx].grid(True, linestyle='--', alpha=0.7)
    
    # Ограничение оси Y для Коши (иначе из-за тяжелых хвостов ящик не видно)
    if name == 'Коши':
        axes[idx].set_ylim(-20, 20)

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300)
plt.show()

# Формирование и вывод таблицы
df_results = pd.DataFrame.from_dict(
    results, 
    orient='index', 
    columns=['Доля выбросов (n=20)', 'Доля выбросов (n=100)', 'Теоретическая доля']
)
print(df_results.round(4))