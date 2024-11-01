from scipy.optimize import minimize, differential_evolution, basinhopping
from abc import abstractmethod
from skopt import gp_minimize
import numpy as np


class LossFunction(object):
    """An abstract class for an arbitrary loss (cost) function.
    This type of function appears frequently in estimation problems where
    the best estimator (given a set of observed data) is the one which
    minimizes some sort of objective function.
    """

    @abstractmethod
    def evaluate(self, params):
        """
        Returns the loss function evaluated at params

        Parameters
        ----------
        params : ndarray
            parameter vector of the model

        Returns
        -------
        loss_fun : scalar
            Returns the scalar value of the loss function evaluated at
            **params**
        """
        pass

    def __call__(self, params):
        """Calls :func:`evaluate`"""
        return self.evaluate(params)

    def fit(self, optimizer="minimize", **kwargs):
        """
        Minimizes the :func:`evaluate` function using :func:`scipy.optimize.minimize`,
        :func:`scipy.optimize.differential_evolution`,
        :func:`scipy.optimize.basinhopping`, or :func:`skopt.gp.gp_minimize`.

        Parameters
        ----------
        optimizer : str
            Optimization algorithm. Options are::

                - ``'minimize'`` uses :func:`scipy.optimize.minimize`

                - ``'differential_evolution'`` uses :func:`scipy.optimize.differential_evolution`

                - ``'basinhopping'`` uses :func:`scipy.optimize.basinhopping`

                - ``'gp_minimize'`` uses :func:`skopt.gp.gp_minimize`

            `'minimize'` is usually robust enough and therefore recommended
            whenever a good initial guess can be provided. The remaining options
            are global optimizers which might provide better results precisely
            in cases where a close engouh initial guess cannot be obtained
            trivially.
        kwargs : dict
            Dictionary for additional arguments.

        Returns
        -------
        opt_result : :class:`scipy.optimize.OptimizeResult` object
            Object containing the results of the optimization process.
            Note: this is also stored in **self.opt_result**.
        """

        if optimizer == "minimize":
            self.opt_result = minimize(self.evaluate, **kwargs)
        elif optimizer == "differential_evolution":
            self.opt_result = differential_evolution(self.evaluate, **kwargs)
        elif optimizer == "basinhopping":
            self.opt_result = basinhopping(self.evaluate, **kwargs)
        elif optimizer == "gp_minimize":
            self.opt_result = gp_minimize(self.evaluate, **kwargs)
        else:
            raise ValueError("optimizer {} is not available".format(optimizer))

        return self.opt_result

    def gradient(self, params):
        """
        Returns the gradient of the loss function evaluated at ``params``

        Parameters
        ----------
        params : ndarray
            parameter vector of the model
        """
        pass

    def hessian(self, params):
        """
        Returns the Hessian matrix of the loss function evaluated at ``params``

        Parameters
        ----------
        params : ndarray
            parameter vector of the model
        """
        raise NotImplementedError

    """
    A base class for a prior distribution. Differently from Likelihood, a prior
    is a PDF that depends solely on the parameters, not on the observed data.
    """


class Prior(LossFunction):
    @property
    def name(self):
        """A name associated with the prior"""
        return self._name

    @name.setter
    def name(self, value="param_name"):
        self._name = value

    @abstractmethod
    def evaluate(self, params):
        """Evaluates the negative of the log of the PDF at ``params``

        Parameters
        ----------
        params : scalar or array-like
            Value at which the PDF will be evaluated

        Returns
        -------
        value : scalar
            Value of the negative of the log of the PDF at params
        """
        pass


class UniformPrior(Prior):
    """Computes the negative log pdf for a n-dimensional independent uniform
    distribution.

    Attributes
    ----------
    lb : int or array-like of ints
        Lower bounds (inclusive)
    ub : int or array-like of ints
        Upper bounds (exclusive)

    Examples
    --------
    >>> from oktopus import UniformPrior
    >>> unif = UniformPrior(0., 1.)
    >>> unif(.5)
    -0.0
    >>> unif(1)
    inf
    """

    def __init__(self, lb, ub, name=None):
        self.lb = np.asarray([lb]).reshape(-1)
        self.ub = np.asarray([ub]).reshape(-1)
        if (self.lb >= self.ub).any():
            raise ValueError(
                "The lower bounds should be smaller than the upper bounds."
            )
        self.name = name

    def __repr__(self):
        return "<UniformPrior(lb={}, ub={})>".format(self.lb, self.ub)

    @property
    def mean(self):
        """Returns the mean of the uniform distributions"""
        return 0.5 * (self.lb + self.ub)

    @property
    def variance(self):
        """Returns the variance of the uniform distributions"""
        return (self.ub - self.lb) ** 2 / 12.0

    def evaluate(self, params):
        if (self.lb <= params).all() and (params < self.ub).all():
            return -np.log(1.0 / (self.ub - self.lb)).sum()
        return np.inf

    def gradient(self, params):
        if (self.lb <= params).all() and (params < self.ub).all():
            return 0.0
        return np.inf


class GaussianPrior(Prior):
    """Computes the negative log pdf for a n-dimensional independent Gaussian
    distribution.

    Attributes
    ----------
    mean : scalar or array-like
        Mean
    var : scalar or array-like
        Variance

    Examples
    --------
    >>> from oktopus import GaussianPrior
    >>> prior = GaussianPrior(0, 1)
    >>> prior(2.)
    2.0
    """

    def __init__(self, mean, var, name=None):
        self.mean = np.asarray([mean]).reshape(-1)
        self.var = np.asarray([var]).reshape(-1)
        self.name = name

    def __repr__(self):
        return "<GaussianPrior(mean={}, var={})>".format(self.mean, self.var)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value

    @property
    def variance(self):
        return self.var

    def evaluate(self, params):
        return ((params - self.mean) ** 2 / (2 * self.var)).sum()

    def gradient(self, params):
        return ((params - self.mean) / self.var).sum()
