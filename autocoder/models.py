import torch
from torch import nn
import math

"""
Все тензоры в задании имеют тип данных float32.
"""

class AE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.encoder = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Генерирует код по объектам.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: Tensor - матрица размера n x d.
        """
        # ваш код здесь
        return self.encoder(x)

    def decode(self, z):
        """
        По матрице латентных представлений z возвращает матрицу объектов x.
        Вход: z, Tensor - матрица n x d латентных представлений.
        Возвращаемое значение: Tensor, матрица объектов n x D.
        """
        # ваш код здесь
        return self.decoder(z)

    def batch_loss(self, batch):
        """
        Вычисляет функцию потерь по батчу - усреднение функции потерь
        по объектам батча.
        Функция потерь по объекту- сумма L2-ошибки восстановления по батчу и
        L2 регуляризации скрытых представлений с весом 1.
        Возвращаемое значение должно быть дифференцируемо по параметрам модели (!).
        Вход: batch, Tensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - функция потерь по батчу.
        """
        # ваш код здесь
        batch_size = batch.shape[0]
        compute_loss = nn.MSELoss()
        z = self.encode(batch)
        l2_reg = torch.sum(z**2)
        output = self.decode(z)
        loss = torch.sum((output - batch)**2, dim=-1).mean(dim=0) + l2_reg
        return loss

    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы объектов x. Использует стандартное нормальное
        распределение в пространстве представлений.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        # ваш код здесь
        mu = torch.zeros(num_samples, self.D)
        sigma = torch.ones(num_samples, self.D)
        x = torch.normal(mu, sigma)
        return self.decode(self.encode(x))
            


def log_mean_exp(data):
    """
    Возвращает логарифм среднего по последнему измерению от экспоненты данной матрицы.
    Подсказка: не забывайте про вычислительную стабильность!
    Вход: mtx, Tensor - тензор размера n_1 x n_2 x ... x n_K.
    Возвращаемое значение: Tensor, тензор размера n_1 x n_2 x ,,, x n_{K - 1}.
    """
    # ваш код здесь
    n = data.shape[-1]
    output = torch.logsumexp(data, dim=-1) - torch.log(torch.tensor(n).float())
    return output


def log_likelihood(x_true, x_distr):
    """
    Вычисляет логарфм правдоподобия объектов x_true для индуцированного
    моделью покомпонентного распределения Бернулли.
    Каждому объекту из x_true соответствуют K сэмплированных распределений
    на x из x_distr.
    Требуется вычислить оценку логарифма правдоподобия для каждого объекта.
    Подсказка: не забывайте про вычислительную стабильность!
    Подсказка: делить логарифм правдоподобия на число компонент объекта не надо.

    Вход: x_true, Tensor - матрица объектов размера n x D.
    Вход: x_distr, Tensor - тензор параметров распределений Бернулли
    размера n x K x D.
    Выход: Tensor, матрица размера n x K - оценки логарифма правдоподобия
    каждого сэмпла.
    """
    # ваш код здесь
    log_l = x_true[:,None,:] * torch.log(x_distr) + (
        1 - x_true[:, None, :]) * torch.log(1 - x_distr)
    return log_l.sum(dim=-1)


def kl(q_distr, p_distr):
    """
    Вычисляется KL-дивергенция KL(q || p) между n парами гауссиан.
    Вход: q_distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: p_distr, tuple(Tensor, Tensor). Аналогично.
    Возвращаемое значение: Tensor, вектор размерности n, каждое значение которого - 
    - KL-дивергенция между соответствующей парой распределений.
    """
    p_mu, p_sigma = p_distr
    q_mu, q_sigma = q_distr
    # ваш код здесь
    n, d = p_mu.shape
    kulbak = torch.log(p_sigma/q_sigma) - 0.5 + (q_sigma**2 + (q_mu - p_mu)**2) / (2 * p_sigma**2)
    kulbak = torch.sum(kulbak, dim=1)
    return kulbak


class VAE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.proposal_network = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
        )
        self.proposal_mu_head = nn.Linear(200, self.d)
        self.proposal_sigma_head = nn.Linear(200, self.d)
        self.generative_network = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def proposal_distr(self, x):
        """
        Генерирует предложное распределение на z.
        Подсказка: областью значений sigma должны быть положительные числа.
        Для этого при генерации sigma следует использовать softplus (!) в качестве
        последнего преобразования.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        # ваш код здесь
        temp = self.proposal_network(x)
        mu = self.proposal_mu_head(temp)
        sigma = self.proposal_sigma_head(temp)
        sigma = nn.Softplus()(sigma)
        return (mu, sigma)

    def prior_distr(self, n):
        """
        Генерирует априорное распределение на z.
        Вход: n, int - число распределений.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        # ваш код здесь
        mu = torch.zeros(n, self.d)
        sigma = torch.ones(n, self.d)
        return mu, sigma

    def sample_latent(self, distr, K=1):
        """
        Генерирует сэмплы из гауссовского распределения на z.
        Сэмплы должны быть дифференцируемы по параметрам распределения!
        Вход: distr, tuple(Tensor, Tensor). Каждое Tensor - матрица размера n x d.
        Первое - mu, второе - sigma.
        Вход: K, int - число сэмплов для каждого объекта.
        Возвращаемое значение: Tensor, матрица размера n x K x d.
        """
        # ваш код здесь
        mu, sigma = distr
        n, d = sigma.size()
        samples = torch.randn((n, K, d))
        samples = mu[:, None, :] + sigma[:, None, :] * samples
        return samples

    def generative_distr(self, z):
        """
        По матрице латентных представлений z возвращает матрицу параметров
        распределения Бернулли для сэмплирования объектов x.
        Вход: z, Tensor - тензор n x K x d латентных представлений.
        Возвращаемое значение: Tensor, тензор параметров распределения
        Бернулли размера n x K x D.
        """
        # ваш код здесь
        x = self.generative_network(z)
        eps = 1e-4
        return torch.clamp(x, eps, 1-eps)
    
    def batch_loss(self, batch):
        """
        Вычисляет вариационную нижнюю оценку логарифма правдоподобия по батчу.
        Вариационная нижняя оценка должна быть дифференцируема по параметрам модели (!),
        т. е. надо использовать репараметризацию.
        Требуется вернуть усреднение вариационных нижних оценок объектов батча.
        Вход: batch, FloatTensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - вариационная нижняя оценка логарифма
        правдоподобия по батчу.
        """
        # ваш код здесь
        batch_size = batch.shape[0]
        q = self.proposal_distr(batch)
        z = self.sample_latent(q)
        p = self.generative_distr(z)
        log_p = log_likelihood(batch, p)
        rec_loss = log_mean_exp(log_p)
        kulbak = kl(q, self.prior_distr(batch_size))
        return (rec_loss - kulbak).mean()

    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы из индуцируемого моделью распределения на объекты x.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        # ваш код здесь
        mu, sigma = self.prior_distr(num_samples)
        z = torch.normal(mu, sigma)
        x = self.generative_network(z)
        return x


def gaussian_log_pdf(distr, samples):
    """
    Функция вычисляет логарифм плотности вероятности в точке относительно соответствующего
    нормального распределения, заданного покомпонентно своими средним и среднеквадратичным отклонением.
    Вход: distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: samples, Tensor - тензор размера n x K x d сэмплов в скрытом пространстве.
    Возвращаемое значение: Tensor, матрица размера n x K, каждый элемент которой - логарифм
    плотности вероятности точки относительно соответствующего распределения.
    """
    mu, sigma = distr
    # ваш код здесь
    c = - 0.5 * torch.log(torch.tensor(2 * math.pi))
    tmp = -1 * (torch.log(sigma[:, None, :]) + (samples - mu[:, None, :])**2/(2 * sigma[:, None, :]**2))
    return (c + tmp).sum(dim=-1)
    

def compute_log_likelihood_monte_carlo(batch, model, K):
    """
    Функция, оценку логарифма правдоподобия вероятностной модели по батчу методом Монте-Карло.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    # ваш код здесь
    batch_size = batch.shape[0]
    prior_z = model.prior_distr(batch_size)
    samples = model.sample_latent(prior_z, K)
    p = model.generative_distr(samples)
    log_p = log_likelihood(batch, p)
    output = log_mean_exp(log_p).mean()
    return float(output)


def compute_log_likelihood_iwae(batch, model, K):
    """
    Функция, оценку IWAE логарифма правдоподобия вероятностной модели по батчу.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, proposal_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    # ваш код здесь
    batch_size = batch.shape[0]
    prior_z = model.prior_distr(batch_size)
    q = model.proposal_distr(batch)
    samples = model.sample_latent(q, K)
    p = model.generative_distr(samples)
    log_p = log_likelihood(batch, p)
    log_prior = gaussian_log_pdf(prior_z, samples)
    log_q = gaussian_log_pdf(q, samples)
    log_total = log_p + log_prior - log_q
    output = log_mean_exp(log_total).mean()
    return float(output)