import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)  # ensure we are not forcing tf_keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter all but ERROR
import pickle
import tempfile
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.mixture import GaussianMixture  # BayesianGaussianMixture
from sklearn.preprocessing import RobustScaler
from typing import Optional, Union, List
from rinet.utils import *

import logging
import absl.logging
absl.logging._warn_preinit_stderr = 0
logging.root.removeHandler(absl.logging._absl_handler)


class RINetV1Pipeline:
    def __init__(
        self, 
        feature_grid_range: List[int] = [-4, 4],
        feature_grid_nbins: int = 100,
    ):
        self.feature_grid_range = feature_grid_range
        self.feature_grid_nbins = feature_grid_nbins
        self.model = None
        self.scaler = None
        self.quantiles = None  # will store the list of quantiles the model outputs
        
    def _build_model(
        self,
    ):
        """
        Define convolutional neural network regressor architecture
        """
        
        input_dim = (self.feature_grid_nbins, 1)
        
        # set of quantiles + reference fraction
        # original quantiles: 
        #    0.01, 0.025, 0.05, 0.10, 
        #    0.20, 0.30,  0.40, 0.50, 
        #    0.60, 0.70,  0.80, 0.90, 
        #    0.95, 0.975, 0.99
        output_dim = 16
            
        inputs = tf.keras.layers.Input(shape=input_dim)
        x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(64, kernel_size=7, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        o = tf.keras.layers.Dense(output_dim, activation='linear')(x)
        model = tf.keras.models.Model(inputs, o)
        self.model = model

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        quantiles: List[int] = [
            0.01, 0.025, 0.05, 0.10,
            0.20, 0.30,  0.40, 0.50,
            0.60, 0.70,  0.80, 0.90,
            0.95, 0.975, 0.99
        ],
        model_save_path: str='./saved_model/',
        loss_fn=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(),
        epochs=20,
        batch_size=8,
    ):
        """
        Fit the model on training data

        Args:
            x_train: list of 1D mixture distribution samples
            y_train: target output statistics:
                [list of quantiles] + [reference fraction]
                In original model:
                    0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 
                    0.40, 0.50,  0.60, 0.70, 0.80, 0.90, 
                    0.95, 0.975, 0.99
            x_val: see above
            y_val: see above
            quantiles: list of quantiles
        """

        assert (y_train.shape[1] - 1) == len(quantiles), "Error: y_train length - 1 doesn't matching length of quantiles"
        assert (y_val.shape[1] - 1) == len(quantiles), "Error: y_val length - 1 doesn't matching length of quantiles"

        self.quantiles = quantiles
        
        # get model
        if self.model is None:
            self._build_model()
        model = self.model
        
        # compile
        model.compile(loss=loss_fn, optimizer=optimizer)

        # prepare features
        x_train = [self._extract_features(i) for i in x_train]
        x_train = np.array(x_train).squeeze()
        x_val = [self._extract_features(i) for i in x_val]
        x_val = np.array(x_val).squeeze()

        # standardize outputs
        scaler = RobustScaler()
        y_train = scaler.fit_transform(y_train)
        y_val = scaler.transform(y_val)
        self.scaler = scaler

        # add channel axis
        x_train = x_train[..., np.newaxis]
        x_val = x_val[..., np.newaxis]

        # train
        # define best model checkpoint
        checkpoint_callback = ModelCheckpoint(
            model_save_path + 'model.weights.h5',
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        
        # train
        history = model.fit(
            x_train, 
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback],
            verbose=2
        )

        # save training history
        with open(model_save_path+'/model_history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # save scaler
        with open(model_save_path+'/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        with open(model_save_path+'/quantiles.pkl', 'wb') as f:
            pickle.dump(quantiles, f)
            
        # load best model
        self.load(model_save_path)

    def load(
        self,
        model_save_path
    ):
        
        if not self.model:
            self._build_model()
        self.model.load_weights(model_save_path + '/model.weights.h5')
        
        scaler = pickle.load(open(model_save_path+'/scaler.pkl', 'rb'))
        self.scaler = scaler

        quantiles = pickle.load(open(model_save_path+'/quantiles.pkl', 'rb'))
        self.quantiles = quantiles
        
    def _extract_features(
        self,
        data: np.ndarray
    ) -> np.array:
        """
        Extracts normalized histogram features from standardized 1D data.
        
        Arguments:
            data: 1D numpy array (n_samples,)
        Returns:
            features: 1D feature vector (normalized histogram)
        """
        
        data = np.asarray(data).squeeze()

        assert data.ndim == 1, 'Error: unexpected shape, input data should be 1D (n_samples,)'
        
        # check for standardized data
        assert np.abs(0 - data.mean()) < 1e-2, 'Error: data not standardized'
        assert np.abs(1 - data.std()) < 1e-2, 'Error: data not standardized'
        
        # histogram
        features = np.histogram(
            data,
            bins=np.linspace(
                self.feature_grid_range[0],
                self.feature_grid_range[1],
                self.feature_grid_nbins+1
            ),
            density=True
        )[0]
        
        # normalize features to [0, 1]
        features = (features - features.min()) / (features.max() - features.min())
        
        return features

    def predict(self, data, log_scale=False, verbose=0):
        '''
        Use model to make a prediction
        
        Args:
            data:     list of 1D numpy arrays
            verbose:  whether to show progress bar
        Returns:
            result_dict: dictionary with predicted statistics
        '''

        if not isinstance(data, list):
            data = [data]

        if log_scale:
            data = [np.log(i) for i in data]
            
        # standardize
        means = [i.mean(axis=0) for i in data]
        stds = [i.std(axis=0) for i in data]
        data = [(i - means[c]) / stds[c] for c, i in enumerate(data)]
    
        # extract features per sample
        feats = [self._extract_features(sample) for sample in data]
        feats = np.stack(feats, axis=0)
        feats = feats[..., np.newaxis]  # add channels axis for CNN-style models
    
        # model prediction
        p = self.model.predict(feats, verbose=verbose)
        p = self.scaler.inverse_transform(p)
    
        results = []
        for i, sample_pred in enumerate(p):
            
            # unscale
            sample_pred[:-1] = (sample_pred[:-1] * stds[i]) + means[i]

            if log_scale:
                sample_pred[:-1] = np.exp(sample_pred[:-1])
            
            result = {
                i: j for i, j in zip(self.quantiles, sample_pred[:-1])
            }
            result['reference_fraction'] = sample_pred[-1]
    
            results.append(result)
    
        # return a single dict if input was a single sample
        return results

        
class RINetV2Pipeline:
    def __init__(
        self, 
        ndim: int = 1,
        feature_grid_range: List[int] = [-4, 4],
        feature_grid_nbins: int = 100,
    ):
        self.ndim = ndim
        self.feature_grid_range = feature_grid_range
        self.feature_grid_nbins = feature_grid_nbins
        self.model = None
        self.scaler = None
        
        assert self.ndim in [1, 2], "Error: ndim should be 1 or 2"
        if self.ndim == 1:
            self.conv_layer = tf.keras.layers.Conv1D
            self.pooling_layer = tf.keras.layers.MaxPooling1D
        else:
            self.conv_layer = tf.keras.layers.Conv2D
            self.pooling_layer = tf.keras.layers.MaxPooling2D
        
    def _build_model(
        self,
    ):
        """
        Define convolutional neural network regressor architecture
        """
        
        if self.ndim == 1:
            input_dim = (self.feature_grid_nbins, 1)
            # mean, std, reference fraction
            output_dim = 3 
        else:
            input_dim = (self.feature_grid_nbins, self.feature_grid_nbins, 1)
            # mean 1, mean 2, std 1, std 2, correlation coefficient, ref. frac.
            output_dim = 6
            
        inputs = tf.keras.layers.Input(shape=input_dim)
        x = self.conv_layer(32, kernel_size=10, activation='relu')(inputs)
        x = self.pooling_layer(pool_size=2)(x)
        x = self.conv_layer(64, kernel_size=10, activation='relu')(x)
        x = self.pooling_layer(pool_size=2)(x)
        x = self.conv_layer(64, kernel_size=10, activation='relu')(x)
        x = self.pooling_layer(pool_size=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        o = tf.keras.layers.Dense(output_dim, activation='linear')(x)
        model = tf.keras.models.Model(inputs, o)
        self.model = model

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        model_save_path
    ):
        """
        Fit the model on training data

        Args:
            x_train: list of 1D or 2D mixture distribution samples
            y_train: target output statistics:
                1D:
                    mean, std, reference fraction
                2D:
                    mean 1, mean 2, std 1, std 2, correlation coeff., reference fraction
            x_val: see above
            y_val: see above
            rest are explanatory
        """
        # get model
        if self.model is None:
            self._build_model()
        model = self.model
        
        # compile
        model.compile(loss = loss_fn, optimizer = optimizer, metrics=['mae', 'mse'])

        # prepare features
        x_train = [self._extract_features(i) for i in x_train]
        x_train = np.array(x_train).squeeze()
        x_val = [self._extract_features(i) for i in x_val]
        x_val = np.array(x_val).squeeze()

        # standardize outputs
        scaler = RobustScaler()
        y_train = scaler.fit_transform(y_train)
        y_val = scaler.transform(y_val)
        self.scaler = scaler

        # add channel axis
        x_train = x_train[..., np.newaxis]
        x_val = x_val[..., np.newaxis]

        # train
        # define best model checkpoint
        checkpoint_callback = ModelCheckpoint(
            model_save_path + 'model.weights.h5',
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_format='tf',
            verbose=1
        )
        
        # train
        history = model.fit(
            x_train, 
            y_train,
            validation_data = (x_val, y_val),
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [checkpoint_callback],
            verbose = 2
        )

        # save training history
        with open(model_save_path+'/model_history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # save scaler
        with open(model_save_path+'/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # load best model
        self.load(model_save_path)

    def load(
        self,
        model_save_path,
        model_file=None
    ):
        if model_file is None:
            if not self.model:
                self._build_model()
            self.model.load_weights(model_save_path + '/model.weights.h5')
        else:
            self.model = keras.models.load_model(f"{model_save_path}/{model_file}")
        
        scaler = pickle.load(open(model_save_path+'/scaler.pkl', 'rb'))
        self.scaler = scaler

    def _extract_features(
        self,
        data: np.ndarray
    ) -> np.array:
        """
        Extracts normalized histogram features from standardized 1D or 2D data.
        
        Arguments:
            data: 1D or 2D numpy array
                  - if 1D: shape (n_samples,)
                  - if 2D: shape (n_samples, 2)
        Returns:
            features: 1D or 2D feature vector (normalized histogram)
        """
        
        data = np.asarray(data).squeeze()
        
        # 1d case
        if data.ndim == 1:
            
            # check for standardized data
            assert np.abs(0 - data.mean()) < 1e-2, 'Error: data not standardized'
            assert np.abs(1 - data.std()) < 1e-2, 'Error: data not standardized'
            
            # histogram
            features = np.histogram(
                data,
                bins=np.linspace(
                    self.feature_grid_range[0], 
                    self.feature_grid_range[1], 
                    self.feature_grid_nbins+1
                ),
                density=True
            )[0]
        
        # 2d case
        elif data.ndim == 2 and data.shape[1] == 2:
            # check for standardized data
            assert np.allclose(data.mean(axis=0), 0, atol=1e-2), 'Error: data not standardized'
            assert np.allclose(data.std(axis=0), 1, atol=1e-2), 'Error: data not standardized'
            
            # 2d histogram
            features = np.histogram2d(
                data[:,0], data[:,1],
                bins=np.linspace(
                    self.feature_grid_range[0], 
                    self.feature_grid_range[1], 
                    self.feature_grid_nbins+1
                ),
                density=True
            )[0]
        
        else:
            raise ValueError("Input must be standardized 1D array or 2D array with 2 features")
        
        # normalize features to [0,1]
        features = (features - features.min()) / (features.max() - features.min())
        
        return features

    def predict(self, data, log_scale=False, verbose=0):
        '''
        Use model to make a prediction
        
        Args:
            data:     list of 1D or 2D numpy arrays
            verbose:  whether to show progress bar
        Returns:
            result_dict: dictionary with predicted statistics
        '''

        if not isinstance(data, list):
            data = [data]

        if log_scale:
            data = [np.log(i) for i in data]
            
        # standardize
        data = [i.reshape(-1, 1) if len(i.shape) == 1 else i for i in data]  # ensure features dimension for 1D inputs
        means = [i.mean(axis=0) for i in data]
        stds = [i.std(axis=0) for i in data]
        data = [(i - means[c]) / stds[c] for c, i in enumerate(data)]
    
        # extract features per sample
        feats = [self._extract_features(sample) for sample in data]
        feats = np.stack(feats, axis=0)
        feats = feats[..., np.newaxis]  # add channels axis for CNN-style models
    
        # model prediction
        p = self.model.predict(feats, verbose=verbose)
        p = self.scaler.inverse_transform(p)
    
        results = []
        for i, sample_pred in enumerate(p):
            if self.ndim == 1:
                p_mean = sample_pred[0]
                p_std  = sample_pred[1]
            
                scaled_mean = (p_mean * stds[i]) + means[i]   # scalar
                scaled_std  = (p_std * stds[i])               # scalar
                        
                cov = correlation_to_covariance(np.array([[1]]), scaled_std)  # (1,1)
            
                result = {
                    'mean': scaled_mean,        # (1,)
                    'covariance': cov,          # (1,1)
                    'std': scaled_std,          # (1,)
                    'correlation': np.nan,
                    'reference_fraction': sample_pred[2] if sample_pred.shape[0] > 2 else np.nan,
                }
    
            elif self.ndim == 2:
                p_mean = sample_pred[:2]
                p_std = sample_pred[2:4]
                p_cor = sample_pred[4]
                scaled_mean = (p_mean * stds[i]) + means[i]
                scaled_std = p_std * stds[i]
    
                cov = correlation_to_covariance(
                    np.array([[1, p_cor], [p_cor, 1]]), scaled_std
                )
    
                result = {
                    'mean': scaled_mean,
                    'covariance': cov,
                    'std': scaled_std,
                    'correlation': p_cor,
                    'reference_fraction': sample_pred[5] if sample_pred.shape[0] > 5 else np.nan
                }
    
            results.append(result)
    
        # return a single dict if input was a single sample
        return results


class GMMPipeline:
    def __init__(self, ndim: int = 1):
        """
        Parameters
        ----------
        ndim : int
            Number of features per point (1 for univariate, 2+ for multivariate).
        """
        self.ndim = ndim
        if ndim < 1:
            raise ValueError("ndim must be >= 1")

    def predict(
        self,
        data: Union[np.ndarray, List],
        max_n_components: int = 4, 
        init_weights: bool = True, 
        bic_logic: bool = False,
        target_n_components: Optional[int] = None,
        n_jobs: int = -1  # parallel processing
    ):
        """
        Fit GMM(s) to either:
          - a single dataset: shape (n_points, ndim)
          - a batch of datasets: shape (batch_size, n_points, ndim)
        """
        if not isinstance(data, list):
            data = [data]
    
        if data[0].ndim == 1:
            data = [i.reshape(-1, 1) for i in data]
    
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._fit_one)(
                d,
                max_n_components,
                init_weights,
                target_n_components,
                bic_logic
            )
            for d in tqdm(data, leave=False)
        )
            
        return results

    
    def _fit_one(
        self, 
        data, 
        max_n_components, 
        init_weights, 
        target_n_components,
        bic_logic = False,
    ):
        fits, bics = [], []
        for n_components in range(1, max_n_components + 1):
            weights_init = None
            if n_components > 1 and init_weights:
                weights_init = [0.55, *(0.45 / (n_components - 1),) * (n_components - 1)]

            gm = GaussianMixture(
                n_components=n_components,
                weights_init=weights_init,
                max_iter=1000,
                random_state=0
            ).fit(data)

            fits.append(gm)
            bics.append(gm.bic(data))

        if target_n_components is not None:
            gm = fits[target_n_components-1]
        elif bic_logic:
            gm = fits[self.best_fit(bics, 0.02)-1]
        else:
            gm = fits[np.argmin(bics)]  # get best fit based on BIC

        best_component = np.argmax(gm.weights_)

        # statistics of best component
        mean = gm.means_[best_component]
        cov = gm.covariances_[best_component]

        # std from covariance
        std = np.sqrt(np.diag(cov))

        # correlation matrix
        denom = np.outer(std, std)

        if data.shape[1] == 1:
            correlation = np.nan
        else:
            correlation = cov / denom
            correlation[denom == 0] = 0.0

        # reference fraction: fraction of points predicted to belong to this component
        labels = gm.predict(data)
        reference_fraction = np.mean(labels == best_component)

        return {
            'mean': mean,
            'covariance': cov,
            'std': std,
            'correlation': correlation,
            'reference_fraction': reference_fraction,
            'gmm_model': gm
        } 

    def best_fit(self, bics, improvement_per_component=0.02):
        """ Determines an optimal fit based on BIC with a penalty for 
            adding more components
        """
        best_fit = 1 # the number of components of the best fit, assume 1 at start
        best_bic = float('inf') # the BIC of the best fit, assume infinity
        for i in range(len(bics)): # loop number of components
            # if the % change of the current BIC from the best BIC
            # is greater than 
            # improvement_per_component * number of additional compnents
            # it is the new best fit
            if (1-bics[i]/best_bic) > improvement_per_component*(i+1-best_fit):
                best_bic = bics[i]
                best_fit = i+1
            else:
                continue
        return best_fit
        

class RefineRPipeline:
    def __init__(self):
        pass

    def predict(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        model: str = 'BoxCox',
        fix_lambda: bool = False,
        n_jobs: int = -1,
        inner_boot: bool = False,
        n_boot: int = 200,
    ):
        """
        Fit reference distribution(s) using refineR in R.
    
        Parameters
        ----------
        data : array-like or list of arrays
            Single dataset: shape (n_points,)
            Batch of datasets: shape (batch_size, n_points)

        model: string model type from refineR FindRI function

        fix_lambda: boolean indicating whether to fix lambda search to 1 (Gaussian)
    
        n_jobs : int, default=-1
            Number of parallel jobs. -1 uses all CPUs.
    
        inner_boot : bool, default=False
            Whether to run refineR's internal bootstrapping (NBootstrap > 0).
            If True, additional bootstrap draws and RI bounds will be returned.
    
        n_boot : int, default=200
            Number of bootstrap replicates to run when inner_boot is True.
    
        Returns
        -------
        dict or list of dict
            For each dataset, a dictionary with:
                - mean, covariance, std, correlation, reference_fraction, lambda, shift
                - ri_low, ri_high (0.025 and 0.975 percentiles)
            If inner_boot=True, also includes:
                - bootstrap_params : structured array of parameter draws
                - bootstrap_ri : structured array of RI bounds
        """
        if not isinstance(data, list):
            data = [data]

        fix_lambda_r = "TRUE" if fix_lambda else "FALSE"

        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._fit_one)(
                d, 
                model=model,
                fix_lambda=fix_lambda_r,
                inner_boot=inner_boot, 
                n_boot=n_boot
            ) for d in data
        )

        return results
        
    def _fit_one(
        self,
        data: np.ndarray,
        model="BoxCox",
        fix_lambda="FALSE",
        inner_boot=False,
        n_boot=200,
    ):
        data = np.asarray(data).ravel()
    
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            np.savetxt(f.name, data, delimiter=",")
            data_file = f.name
    
        # build the R code
        r_code = f"""
        suppressMessages(library(refineR))
        data <- scan("{data_file}", sep=",", quiet=TRUE)
        fit <- tryCatch(
            findRI(
                data,
                model='{model}',
                warnings=FALSE,
                NBootstrap={n_boot if inner_boot else 0},
                return_boot={str(inner_boot).upper()},
                fix_lambda={fix_lambda}
            ),
            error = function(e) NULL
        )
        if (is.null(fit)) {{
            quit(status=1)
        }}
    
        # core parameters
        mu <- fit$Mu
        sigma <- fit$Sigma
        lambda <- fit$Lambda
        shift <- fit$Shift
        p <- fit$P
    
        # RI point estimates
        results <- getRI(fit)
        low <- results$PointEst[results$Percentile == 0.025]
        high <- results$PointEst[results$Percentile == 0.975]
    
        # if bootstrap, also dump parameters + RI table
        if ({str(inner_boot).upper()}) {{
            boot_params <- fit$BootstrapParams
            boot_ri <- fit$BootstrapRI
            write.csv(boot_params, file="{data_file}.boot_params", row.names=FALSE)
            write.csv(boot_ri, file="{data_file}.boot_ri", row.names=FALSE)
        }}
    
        result <- c(mu, sigma, lambda, shift, p, low, high)
        write.table(result, file="{data_file}.out", sep=",", row.names=FALSE, col.names=FALSE)
        """
    
        r_file = data_file + ".R"
        with open(r_file, "w") as f:
            f.write(r_code)
    
        try:
            subprocess.run(["Rscript", r_file], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # R crashed or refineR failed → return None
            return None
    
        result, boot_params, boot_ri = None, None, None
        try:
            result = np.loadtxt(data_file + ".out", delimiter=",")
            if inner_boot:
                boot_params = np.genfromtxt(data_file + ".boot_params", delimiter=",", names=True)
                boot_ri = np.genfromtxt(data_file + ".boot_ri", delimiter=",", names=True)
        except Exception:
            return None
        finally:
            # Clean up files
            for fpath in [
                data_file,
                r_file,
                data_file + ".out",
                data_file + ".boot_params",
                data_file + ".boot_ri",
            ]:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
    
        if result is None or result.size != 7:
            return None
    
        mu, sigma, lam, shift, p, low, high = result
    
        return {
            "mean": np.array([mu]),
            "covariance": np.array([[sigma]]),
            "std": np.array([np.sqrt(sigma)]),
            "correlation": np.nan,
            "reference_fraction": 1 - p,
            "lambda": lam,
            "shift": shift,
            "ri_low": low,
            "ri_high": high,
            "bootstrap_params": boot_params,
            "bootstrap_ri": boot_ri,
        }


class ReflimRPipeline:
    def __init__(self):
        pass

    def predict(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        n_jobs: int = -1,
    ):
        """
        Fit reference distribution(s) using refineR in R.
    
        Parameters
        ----------
        data : array-like or list of arrays
            Single dataset: shape (n_points,)
            Batch of datasets: shape (batch_size, n_points)
    
        n_jobs : int, default=-1
            Number of parallel jobs. -1 uses all CPUs.
    
        Returns
        -------
        dict or list of dict
            For each dataset, a dictionary with:
                - mean, covariance, std, correlation, reference_fraction, lambda, shift
                - ri_low, ri_high (0.025 and 0.975 percentiles)
            If inner_boot=True, also includes:
                - bootstrap_params : structured array of parameter draws
                - bootstrap_ri : structured array of RI bounds
        """
        if not isinstance(data, list):
            data = [data]

        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._fit_one)(
                d, 
            ) for d in data
        )

        return results
        
    def _fit_one(self, data: np.ndarray):
        data = np.asarray(data).ravel()
    
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            np.savetxt(f.name, data, delimiter=",")
            data_file = f.name
    
        r_code = f"""
        suppressMessages(library(reflimR))
        data <- scan("{data_file}", sep=",", quiet=TRUE)
        fit <- tryCatch(reflim(data, plot.it=FALSE), error=function(e) NULL)
    
        if (is.null(fit)) {{
          quit(status=1)
        }}
    
        is_lognormal_num <- as.integer(fit$lognormal)
    
        if (fit$lognormal) {{
          mu <- fit$stats["meanlog"]
          sd <- fit$stats["sdlog"]
        }} else {{
          mu <- fit$stats["mean"]
          sd <- fit$stats["sd"]
        }}
    
        p <- fit$perc.norm
    
        low <- fit$limits["lower.lim"]
        high <- fit$limits["upper.lim"]
    
        low_ci_low  <- fit$confidence.int["lower.lim.low"]
        low_ci_high <- fit$confidence.int["lower.lim.upp"]
        high_ci_low <- fit$confidence.int["upper.lim.low"]
        high_ci_high<- fit$confidence.int["upper.lim.upp"]
    
        vals <- c(is_lognormal_num, mu, sd, p, low, high,
                  low_ci_low, low_ci_high, high_ci_low, high_ci_high)
    
        cat(paste(vals, collapse=","), file="{data_file}.out")
        """
    
        r_file = data_file + ".R"
        with open(r_file, "w") as f:
            f.write(r_code)
    
        try:
            subprocess.run(["Rscript", r_file], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # R failed → return None
            return None
    
        try:
            vals = np.loadtxt(data_file + ".out", delimiter=",", dtype=float)
        except Exception:
            return None
    
        vals = np.atleast_1d(vals)
        if vals.size != 10:
            return None
    
        is_lognormal = bool(int(vals[0]))
        mu, sd, p, low, high, low_ci_low, low_ci_high, high_ci_low, high_ci_high = vals[1:]
    
        return {
            "mean": np.array([mu]) if not is_lognormal else np.nan,
            "mean_log": np.array([mu]) if is_lognormal else np.nan,
            "std": np.array([sd]) if not is_lognormal else np.nan,
            "std_log": np.array([sd]) if is_lognormal else np.nan,
            "is_lognormal": is_lognormal,
            "reference_fraction": p,
            "ri_low": low,
            "ri_high": high,
            "ri_low_ci": np.array([low_ci_low, low_ci_high]),
            "ri_high_ci": np.array([high_ci_low, high_ci_high]),
        }
    
    
        