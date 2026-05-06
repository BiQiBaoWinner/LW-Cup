import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.DLOB import DLOB
from config import pv_cols, ob_cols_base, ob_cols_derive1, ob_cols_derive2, ob_cols_pro


if __name__ == "__main__":
    extra_cols = ['volume_delta'] + ob_cols_base + ob_cols_derive1 + ob_cols_derive2 + ob_cols_pro

    model = DLOB(
        task_label="label_5",
        use_pool=False,
        extra_cols=extra_cols,
        window=100,
        batch_size=2048,
        lr=1e-2,
        seed=42,
        device='cuda:0',
        n_jobs=16,
    )
    model.fit_model(epochs=120, early_stopping=20)
    model.save_model("dlob_model.pt")
