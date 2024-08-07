import os
import time

from mlc.load_do_save import load_hdf5, save_hdf5
from mlc.models.pipeline import Pipeline


class LazyPipeline:
    def __init__(self, pipeline: Pipeline):
        self.name = pipeline.name
        self.pipeline = pipeline
        self.path = None
        self.loaded = False
        self.pred = None

    # LOADING METHODS
    def _pipeline_load(self):
        if self.path is None:
            raise ValueError("Path is not set")
        if not self.loaded:
            self.pipeline.load(self.path)
            self.loaded = True

    def _pred_load(self):
        if self.path is None:
            raise ValueError("Path is not set")
        if self.pred is None:
            pred_path = f"{self.path}.pred.hdf5"
            while (time.time() - os.path.getmtime("test.time")) < 240:
                time.sleep(10)
                print("Waiting for file.")
            self.pred = load_hdf5(pred_path)

    def load(self, path: str) -> None:
        self.path = path
        self.loaded = False
        self.pred = None

    # SAVING METHODS

    def save_pred(self, x) -> None:
        pred_path = f"{self.path}.pred.hdf5"
        print(pred_path)
        if os.path.exists(pred_path):
            return None

        print("Saving predictions")
        self._pipeline_load()
        if self.pipeline.objective in ["regression"]:
            y_pred = self.pipeline.predict(x)
        elif self.pipeline.objective in ["binary", "classification"]:
            y_pred = self.pipeline.predict_proba(x)
        else:
            raise ValueError("Unknown objective")
        save_hdf5(y_pred, pred_path)

    def save(self, path: str) -> None:
        self.path = path
        self.pipeline.save(path)

    # UTILITY METHODS

    def __getattr__(self, item):
        self._pipeline_load()

        return object.__getattribute__(self.pipeline, item)

    def clone(self):
        return self.pipeline.__class__([step.clone() for step in self.steps])

    def __getitem__(self, item):
        return self.pipeline.steps[item]

    # PREDICTION METHODS]

    def predict(self, x):
        self._pipeline_load()
        return self.pipeline.predict(x)

    def predict_proba(self, x):
        self._pipeline_load()
        return self.pipeline.predict_proba(x)

    def transform(self, x):
        self._pipeline_load()
        return self.pipeline.transform(x)

    def lazy_predict(self, start_idx, end_idx):
        self._pred_load()
        print("Lazy prediction")
        return self.pred[start_idx:end_idx]

    def safe_lazy_predict(self, x, start_idx, end_idx):
        self.save_pred(x)
        return self.lazy_predict(start_idx, end_idx)

    def fit(self, x, y, x_val=None, y_val=None):
        self.pipeline.fit(x, y, x_val, y_val)
