import math
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model
from river.drift import PageHinkley
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from drift_study.drift_detectors.drift_detector import DriftDetector


class PcaCdDrift(DriftDetector):
    """Principal Component Analysis Change Detection (PCA-CD) is a drift
    detection algorithm which checks for change in the distribution of the
    given data using one of several divergence metrics calculated on the data's
    principal components.

    First, principal components are built from the reference window - the
    initial ``window_size`` samples. New samples from the test window, of the
    same width, are projected onto these principal components. The divergence
    metric is calculated on these scores for the reference and test windows; if
    this metric diverges enough, then we consider drift to have occurred. This
    threshold is determined dynamically through the use of the Page-Hinkley
    test.

    Once drift is detected, the reference window is replaced with the current
    test window, and the test window is initialized.

    Ref. :cite:t:`qahtan2015pca`

    Attributes:
        step (int): how frequently (by number of samples), to detect drift.
            This is either 100 samples or ``sample_period * window_size``,
            whichever is smaller.
        ph_threshold (float): threshold parameter for the internal Page-Hinkley
            detector. Takes the value of ``.01 * window_size``.
        num_pcs (int): the number of principal components being used to meet
            the specified ``ev_threshold`` parameter.
    """

    input_type = "batch"

    def __init__(
        self,
        batch_size: int,
        divergence_metric: str = "kl",
        ev_threshold: float = 0.99,
        delta: float = 0.1,
        ph_t_ratio: float = 0.001,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Args:
            window_size (int): size of the reference window. Note that
                ``PCA_CD`` will only try to detect drift periodically, either
                every 100 observations or 5% of the ``window_size``, whichever
                is smaller.
            ev_threshold (float, optional): Threshold for percent explained
                variance required when selecting number of principal
                components.
                Defaults to 0.99.
            delta (float, optional): Parameter for Page Hinkley test. Minimum
                amplitude of change in data needed to sound alarm. Defaults to
                0.1.
            divergence_metric (str, optional): divergence metric when comparing
                the two distributions when detecting drift. Defaults to "kl".

                * "kl" - Jensen-Shannon distance, a symmetric bounded form of
                  Kullback-Leibler divergence, uses kernel density estimation
                  with Epanechnikov kernel

                * "intersection" - intersection area under the curves for the
                  estimated density functions, uses histograms to estimate
                  densities of windows. A discontinuous, less accurate estimate
                  that should only be used when efficiency is of concern.

        """

        super().__init__(
            batch_size=batch_size,
            divergence_metric=divergence_metric,
            ev_threshold=ev_threshold,
            delta=delta,
            ph_t_ratio=ph_t_ratio,
            **kwargs,
        )
        self.window_size = None
        self.ph_threshold = None
        self.bins = None
        self.batch_size = batch_size
        self.ph_t_ratio = ph_t_ratio

        self.ev_threshold = ev_threshold
        self.divergence_metric = divergence_metric

        # Initialize parameters

        self.delta = delta

        self._drift_detection_monitor = None
        self.num_pcs = None

        self._reference_window = pd.DataFrame()
        self._test_window = pd.DataFrame()
        self._pca = None
        self._reference_pca_projection = pd.DataFrame()
        self._test_pca_projection = pd.DataFrame()
        self._density_reference = {}

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:

        self.window_size = len(x)
        self.ph_threshold = (
            round(self.ph_t_ratio * self.window_size) / self.batch_size
        )

        self.bins = int(np.floor(np.sqrt(self.window_size)))
        self._reference_window = x
        self._test_window = x
        self._pca = PCA(self.ev_threshold)
        self._pca.fit(self._reference_window)
        self.num_pcs = len(self._pca.components_)

        self._drift_detection_monitor = PageHinkley(
            delta=self.delta, threshold=self.ph_threshold, min_instances=10
        )

        # Project ref window onto PCs
        self._reference_pca_projection = pd.DataFrame(
            self._pca.transform(self._reference_window),
        )

        # Project test window onto PCs
        self._test_pca_projection = pd.DataFrame(
            self._pca.transform(self._test_window),
        )

        for i in range(self.num_pcs):

            if self.divergence_metric in ["intersection", "mkl"]:
                # Histograms need the same bin edges
                # so find bounds from
                # both windows to inform range for reference and test
                self.lower = min(
                    self._reference_pca_projection.iloc[:, i].min(),
                    self._test_pca_projection.iloc[:, i].min(),
                )

                self.upper = max(
                    self._reference_pca_projection.iloc[:, i].max(),
                    self._test_pca_projection.iloc[:, i].max(),
                )

                self._density_reference[f"PC{i + 1}"] = self._build_histograms(
                    self._reference_pca_projection.iloc[:, i],
                    bins=self.bins,
                    bin_range=(self.lower, self.upper),
                )

            else:
                self._density_reference[f"PC{i + 1}"] = self._build_kde(
                    self._reference_pca_projection.iloc[:, i]
                )

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        x = pd.DataFrame(x)
        self._test_window = pd.concat([self._test_window, x])
        self._test_window = self._test_window.iloc[-self.window_size :]

        # Project new observation onto PCs
        next_proj = pd.DataFrame(
            self._pca.transform(x),
        )

        # Winsorize incoming data to align with
        # reference and test histograms
        if self.divergence_metric in ["intersection", "mkl"]:
            for i in range(self.num_pcs):
                if next_proj.iloc[0, i] < self.lower:
                    next_proj.iloc[0, i] = self.lower

                elif next_proj.iloc[0, i] > self.upper:
                    next_proj.iloc[0, i] = self.upper

        # Add projection to test projection data
        self._test_pca_projection = pd.concat(
            [self._test_pca_projection, next_proj]
        )
        self._test_pca_projection = self._test_pca_projection.iloc[
            -self.window_size :
        ]

        # Compute density distribution for test data
        self._density_test = {}
        for i in range(self.num_pcs):

            if self.divergence_metric in ["intersection", "mkl"]:

                self._density_test[f"PC{i + 1}"] = self._build_histograms(
                    self._test_pca_projection.iloc[:, i],
                    bins=self.bins,
                    bin_range=(self.lower, self.upper),
                )

            elif self.divergence_metric in ["js"]:
                self._density_test[f"PC{i + 1}"] = self._build_kde(
                    self._test_pca_projection.iloc[:, i]
                )

        # Compute current score
        change_scores = []

        if self.divergence_metric == "js":
            for i in range(self.num_pcs):

                change_scores.append(
                    self._jensen_shannon_distance(
                        self._density_reference[f"PC{i + 1}"],
                        self._density_test[f"PC{i + 1}"],
                    )
                )

        elif self.divergence_metric == "mkl":
            for i in range(self.num_pcs):
                change_scores.append(
                    self._max_kl(
                        self._density_reference[f"PC{i + 1}"],
                        self._density_test[f"PC{i + 1}"],
                    )
                )

        elif self.divergence_metric == "intersection":
            for i in range(self.num_pcs):
                change_scores.append(
                    self._intersection_divergence(
                        self._density_reference[f"PC{i + 1}"],
                        self._density_test[f"PC{i + 1}"],
                    )
                )

        change_score = max(change_scores)

        self._drift_detection_monitor.update(change_score)

        return (
            self._drift_detection_monitor.drift_detected,
            False,
            pd.DataFrame({"change_score": [change_score]}),
        )

    @classmethod
    def _build_kde(cls, sample):
        """Compute the Kernel Density Estimate for a given 1D data stream

        Args:
            sample: 1D data for which we desire to estimate its density
                function

        Returns:
            Dict with density estimates for each value and KDE object

        """
        sample_length = len(sample)
        bandwidth = 1.06 * np.std(sample, ddof=1) * (sample_length ** (-1 / 5))
        kde_object = KernelDensity(
            bandwidth=bandwidth, kernel="epanechnikov"
        ).fit(sample.values.reshape(-1, 1))
        # score_samples gives log-likelihood for each point,
        # true density values should be > 0 so exponentiate
        density = np.exp(
            kde_object.score_samples(sample.values.reshape(-1, 1))
        )

        return {"density": density, "object": kde_object}

    @staticmethod
    def _build_histograms(sample, bins, bin_range):
        """
        Compute the histogram density estimates for a given 1D data stream.
        Density estimates consist of the value of the pdf in each bin,
        normalized s.t. integral over the entire range is 1

        Args:
            sample: 1D array in which we desire to estimate its density
                function
            bins: number of bins for estimating histograms. Equal to sqrt of
                cardinality of ref window
            bin_range: (float, float) lower and upper bound of histogram bins

        Returns:
            Dict of bin edges and corresponding density values (normalized s.t.
            they sum to 1)

        """

        density = np.histogram(
            sample, bins=bins, range=bin_range, density=True
        )
        return {
            "bin_edges": list(density[1]),
            "density": list(density[0] / np.sum(density[0])),
        }

    @classmethod
    def _jensen_shannon_distance(cls, density_reference, density_test):
        """Computes Jensen Shannon between two distributions

        Args:
            density_reference (dict): dictionary of density values and object
                from ref distribution
            density_test (dict): dictionary of density values and object from
                test distribution

        Returns:
            Change Score

        """
        js = jensenshannon(
            density_reference["density"], density_test["density"]
        )
        return js

    @classmethod
    def _max_kl(cls, density_reference, density_test):
        """Computes Jensen Shannon between two distributions

        Args:
            density_reference (dict): dictionary of density values and object
                from ref distribution
            density_test (dict): dictionary of density values and object from
                test distribution

        Returns:
            Change Score

        """
        p = np.array(density_reference["density"])
        q = np.array(density_test["density"])

        p_c = p.copy()
        q_c = q.copy()

        q_c[(p != 0) & (q == 0)] = np.finfo(float).eps
        p_c[(q != 0) & (p == 0)] = np.finfo(float).eps

        mkl = max(
            entropy(p, q_c),
            entropy(q, p_c),
        )
        if math.isinf(mkl):
            print("Problem")
        return mkl

    @staticmethod
    def _intersection_divergence(density_reference, density_test):
        """
        Computes Intersection Area similarity between two distributions using
        histogram density estimation method. A value of 0 means the
        distributions are identical, a value of 1 means they are completely
        different

        Args:
            density_reference (dict): dictionary of density values from
                reference distribution
            density_test (dict): dictionary of density values from test
                distribution

        Returns:
            Change score

        """

        intersection = np.sum(
            np.minimum(density_reference["density"], density_test["density"])
        )
        divergence = 1 - intersection

        return divergence

    def needs_model(self) -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "divergence_metric": trial.suggest_categorical(
                "divergence_metric", ["mkl", "intersection"]
            ),
            "ev_threshold": trial.suggest_float("ev_threshold", 0.5, 1 - 1e-6),
            "delta": trial.suggest_float("delta", 1e-4, 0.2),
            "ph_t_ratio": trial.suggest_float("ph_t_ratio", 1e-4, 1e-1),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "divergence_metric": "mkl",
            "ev_threshold": 0.99,
            "delta": 0.1,
            "ph_t_ratio": 0.001,
        }


detectors: Dict[str, Type[DriftDetector]] = {"pca_cd": PcaCdDrift}
