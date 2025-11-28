from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import scipy
from scipy.special import boxcox
import numpy as np

class datascalers_pt():

    def __init__(self):

        torch.set_default_dtype(torch.float32)
        torch.set_printoptions(precision=8)

    def StandardScaler(self, copy=True):
        self.StandardScaler = 0
        self.StandardScaler =  self.StandardScaler_pt(copy=copy)
        return self.StandardScaler

    def BoxCox(self, standardize=True,copy=True):
        self.BoxCox = 0
        self.BoxCox = self.BoxCox_pt(self.StandardScaler_pt(copy=copy),standardize=standardize,copy=copy)
        return self.BoxCox

    def YeoJohnson(self, standardize=True,copy=True):
        self.YeoJohnson = 0
        self.YeoJohnson = self.YeoJohnson_pt(self.StandardScaler_pt(copy=copy),standardize=standardize,copy=copy)
        return self.YeoJohnson
    
    class StandardScaler_pt():

        def __init__(self, copy=True):
            self.copy = copy
            self.mean_standard_scaler = 0
            self.std_standard_scaler = 1

        def _standard_scaler_transform(self, X, copy=True):

            self.copy = copy

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if self.copy:
                X = X.clone()

            if (not self.mean_standard_scaler.nelement()==0 and not self.std_standard_scaler.nelement()==0):
                out = X - self.mean_standard_scaler
                out = out / self.std_standard_scaler
                return out
            else:
                raise ValueError("Mean and/or std of the scaler are not set. Did you run the fit-function already?")

        def standard_scaler_inverse_transform(self, X, copy=True):

            self.copy = copy

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if self.copy:
                X = X.clone()

            out = X * self.std_standard_scaler
            out = out + self.mean_standard_scaler
            return out


        def fit(self, X, copy=True):

            self.copy = copy

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if self.copy:
                X = X.clone()

            #self.mask_fit = ~torch.logical_or(torch.any(X.isnan(),dim=1),torch.any(X.isinf(),dim=1))
            self.mask_fit = ~torch.any(X.isnan(),dim=1)

            self.mean_standard_scaler = X[self.mask_fit].mean(0, keepdim=True)
            self.std_standard_scaler = X[self.mask_fit].std(0, unbiased=False, keepdim=True)


        def transform(self, X, copy=True):

            return self._standard_scaler_transform(X, copy=copy)

    class BoxCox_pt():

        def __init__(self, StandardScaler_pt, standardize=True, copy=True):
            self.copy = copy
            self.standardize = standardize
            self.standard_scaler = StandardScaler_pt

        def _box_cox_inverse_tranform(self, x, lmbda):
            """Return inverse-transformed input x following Box-Cox inverse
            transform with parameter lambda.
            """
            if lmbda == 0:
                x_inv = torch.exp(x)
            else:
                x_inv = torch.pow((x * lmbda + 1), (1 / lmbda))

            return x_inv

        def _box_cox_optimize(self, x):
            """Find and return optimal lambda parameter of the Box-Cox transform by
            MLE, for observed data x.
            We here use scipy builtins which uses the brent optimizer.
            """
            # the computation of lambda is influenced by NaNs so we need to
            # get rid of them
            _, lmbda = scipy.stats.boxcox(x[~torch.isnan(x)], lmbda=None)

            return lmbda

        def _box_cox_transform(self, x, lmbda):

            return boxcox(x, lmbda)
            

        def fit(self, X, standardize=True, copy=True, force_transform=False):

            self.standardize = standardize
            self.copy = copy

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if self.copy:
                X = X.clone()

            with np.errstate(invalid="ignore"):  # hide NaN warnings
                self.lambdas_ = torch.tensor([self._box_cox_optimize(col) for col in X.T])

            if self.standardize or force_transform:

                for i, lmbda in enumerate(self.lambdas_):
                    with np.errstate(invalid="ignore"):  # hide NaN warnings
                        X[:, i] = self._box_cox_transform(X[:, i], lmbda)


            if self.standardize:

                self.standard_scaler.fit(X)

                if force_transform:
                    X = self.standard_scaler.transform(X)

            return X


        def transform(self, X):

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)

            out = torch.zeros_like(X)

            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    out[:, i] = self._box_cox_transform(X[:, i], lmbda)

            if self.standardize:
                out = self.standard_scaler.transform(out)

            return out


        def inverse_transform(self, X):
            
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)

            out = torch.zeros_like(X)

            if self.standardize:
                out = self.standard_scaler.inverse_transform(X)

            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    X[:, i] = self._box_cox_inverse_transform(X[:, i], lmbda)

            return X

    class YeoJohnson_pt():

        def __init__(self, StandardScaler_pt, standardize=True, copy=True):
            self.copy = copy
            self.standardize = standardize
            self.standard_scaler = StandardScaler_pt

        def _yeo_johnson_transform(self, x, lmbda):
            """Return transformed input x following Yeo-Johnson transform with
            parameter lambda.
            """

            mask_pos_x = x>=0
            out = torch.zeros_like(x)

            if np.abs(lmbda) < np.spacing(1.0):
                out[mask_pos_x] = torch.log(x[mask_pos_x]+1.)
            else:
                out[mask_pos_x] = (torch.pow((x[mask_pos_x]+1.),lmbda) - 1.)/lmbda
            if np.abs(lmbda-2.) > np.spacing(1.0):
                out[~mask_pos_x] = -(torch.pow((1.-x[~mask_pos_x]),(2.-lmbda)) - 1.)/(2.-lmbda)
            else:
                out[~mask_pos_x] = -torch.log(1.-x[~mask_pos_x])

            return out

        def _yeo_johnson_optimize(self, x):
            """Find and return optimal lambda parameter of the Yeo-Johnson
            transform by MLE, for observed data x.
            Like for Box-Cox, MLE is done via the brent optimizer.
            """

            def _neg_log_likelihood(lmbda):
                """Return the negative log likelihood of the observed data x as a
                function of lambda."""
                x_trans = self._yeo_johnson_transform(x, lmbda)
                n_samples = x.size()[0]

                loglike = -n_samples / 2. * torch.log(x_trans.var(unbiased=False))
                loglike += (lmbda - 1.) * torch.sum(torch.sign(x) * torch.log1p(torch.abs(x)))

                return -loglike

            # the computation of lambda is influenced by NaNs so we need to
            # get rid of them
            x = x[~torch.isnan(x)]

            return scipy.optimize.brent(_neg_log_likelihood, brack=(-2, 2))

        def _yeo_johnson_inverse_transform(self, x, lmbda):
            """Return inverse-transformed input x following Yeo-Johnson inverse
            transform with parameter lambda.
            """
            x_inv = torch.zeros_like(x)
            mask_pos_x = x >= 0

            # when x >= 0
            if abs(lmbda) < np.spacing(1.0):
                x_inv[mask_pos_x] = torch.exp(x[mask_pos_x]) - 1.
            else:  # lmbda != 0
                x_inv[mask_pos_x] = torch.power(x[mask_pos_x] * lmbda + 1., 1. / lmbda) - 1.

            # when x < 0
            if abs(lmbda - 2) > np.spacing(1.0):
                x_inv[~mask_pos_x] = 1. - torch.power(-(2. - lmbda) * x[~mask_pos_x] + 1., 1. / (2. - lmbda))
            else:  # lmbda == 2
                x_inv[~mask_pos_x] = 1. - torch.exp(-x[~mask_pos_x])

            return x_inv


        def fit(self, X, standardize=True, copy=True, force_transform=False):

            self.standardize = standardize
            self.copy = copy
            self.force_transform = force_transform

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if self.copy:
                X = X.clone()

            with np.errstate(invalid="ignore"):  # hide NaN warnings
                self.lambdas_ = torch.tensor([self._yeo_johnson_optimize(col) for col in X.T])

            if self.standardize or force_transform:

                for i, lmbda in enumerate(self.lambdas_):
                    with np.errstate(invalid="ignore"):  # hide NaN warnings
                        X[:, i] = self._yeo_johnson_transform(X[:, i], lmbda)


            if self.standardize:

                self.standard_scaler.fit(X)

                if force_transform:
                    X = self.standard_scaler.transform(X)

            return X


        def transform(self, X):

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)

            out = torch.zeros_like(X)

            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    out[:, i] = self._yeo_johnson_transform(X[:, i], lmbda)

            if self.standardize:
                out = self.standard_scaler.transform(out)

            return out


        def inverse_transform(self, X):
            
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)

            out = torch.zeros_like(X)

            if self.standardize:
                out = self.standard_scaler.inverse_transform(X)

            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    X[:, i] = self._yeo_johnson_inverse_transform(X[:, i], lmbda)

            return X
