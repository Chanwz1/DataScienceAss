from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency-encodes high-cardinality categorical columns (fit on train only)."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.maps_ = {c: X[c].value_counts(normalize=True) for c in self.cols}
        return self

    def transform(self, X):
        X = X[self.cols].copy()
        for c in self.cols:
            X[c] = X[c].map(self.maps_[c]).fillna(0.0)
        return X.values