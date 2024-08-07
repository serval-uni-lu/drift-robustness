# from typing import Union, Tuple, Type, Optional
#
# import numpy as np
# import numpy.typing as npt
# import pandas as pd
# from frouros.unsupervised.base import UnsupervisedBaseEstimator
# from frouros.unsupervised.statistical_test import KSTest
# from mlc.models.model import Model
#
# from drift_study.drift_detectors.drift_detector import DriftDetector
#
#
# class UnsupervisedFrourosDrift(DriftDetector):
#     def __init__(
#         self,
#         window_size: int,
#         internal_detector_cls: Type[UnsupervisedBaseEstimator],
#         internal_args: Union[dict, None] = None,
#         alpha: float = 0.01,
#         **kwargs,
#     ) -> None:
#         self.internal_detector_cls = internal_detector_cls
#
#         self.internal_args = internal_args
#         if self.internal_args is None:
#             self.internal_args = {}
#         self.drift_detector = None
#         self.window_size = window_size
#         self.x_test = None
#
#     def fit(
#         self,
#         x: pd.DataFrame,
#         t: Union[pd.Series, npt.NDArray[np.int_]],
#         y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
#         y_scores: Union[npt.NDArray[np.float_]],
#         model: Optional[Model],
#     ) -> None:
#         self.drift_detector = self.internal_detector_cls()
#         self.drift_detector.fit(x)
#         self.x_test = x
#
#     def update(
#         self,
#         x: pd.DataFrame,
#         t: Union[pd.Series, npt.NDArray[np.int_]],
#         y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
#         y_scores: Union[npt.NDArray[np.float_]],
#     ) -> Tuple[bool, bool, pd.DataFrame]:
#         x = pd.DataFrame(x)
#         self.x_test = pd.concat([self.x_test, x])
#         self.x_test = self.x_test.iloc[-self.window_size :]
#         self.drift_detector.
#         return False, False, pd.DataFrame()
#
#     @staticmethod
#     def needs_label() -> bool:
#         return False
#
#
# class KSTestDrift(UnsupervisedFrourosDrift):
#     def __init__(
#         self,
#         window_size: int,
#         internal_args: Union[dict, None] = None,
#         **kwargs,
#     ):
#         super().__init__(window_size, KSTest, internal_args, **kwargs)
#
#
# detectors = {
#     "ks_test": KSTestDrift,
# }

# TO IMPLEMENT
