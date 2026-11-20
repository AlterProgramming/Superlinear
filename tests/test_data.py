import numpy as np
import pytest
from multisuper.data import SYNTHETIC,REAL,load_dataset

@pytest.mark.parametrize('name',list(SYNTHETIC)+list(REAL))
def test_disjoint_train_validation_test(name):
    d=load_dataset(name,seed=0)
    assert len(d.X_train)>0 and len(d.X_val)>0 and len(d.X_test)>0
    assert d.X_train.shape[1]==d.X_test.shape[1]
    assert np.isfinite(d.X_train).all() and np.isfinite(d.y_train).all()
    assert d.provenance == ('synthetic' if name in SYNTHETIC else 'real_observational')

@pytest.mark.parametrize('name',SYNTHETIC)
def test_synthetic_reproducibility(name):
    a,b=load_dataset(name,seed=13),load_dataset(name,seed=13)
    np.testing.assert_array_equal(a.X_train,b.X_train)
    np.testing.assert_array_equal(a.y_test,b.y_test)
    assert (a.X_ood is None) if name in ("parity6", "noise_only") else (a.X_ood is not None)


def test_different_synthetic_seeds_change_data():
    a,b=load_dataset('bilinear',seed=0),load_dataset('bilinear',seed=1)
    assert not np.array_equal(a.X_train,b.X_train)
