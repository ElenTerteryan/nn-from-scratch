import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = shuffled_df.iloc[:n_train]
    val_df = shuffled_df.iloc[n_train : n_train + n_val]
    test_df = shuffled_df.iloc[n_train + n_val :]

    return train_df, val_df, test_df


class TargetEncoder:

    def __init__(self) -> None:
        self.category_map: Dict[str, Dict[Any, float]] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str], target_col: str) -> None:
        for col in cat_cols:
            cat2mean: Dict[Any, float] = {}
            grouped = df.groupby(col)[target_col].mean()
            for category_val, avg_target in grouped.items():
                cat2mean[category_val] = float(avg_target)
            self.category_map[col] = cat2mean

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        df_enc = df.copy()
        for col in cat_cols:
            cat2mean = self.category_map[col]
            global_mean = np.mean(list(cat2mean.values()))  
            df_enc[col] = df_enc[col].apply(lambda x: cat2mean.get(x, global_mean))
        return df_enc

    def fit_transform(self, df: pd.DataFrame, cat_cols: List[str], target_col: str) -> pd.DataFrame:
        self.fit(df, cat_cols, target_col)
        return self.transform(df, cat_cols)


class StandardScaler:
 
    def __init__(self) -> None:
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> None:
        for col in cols:
            self.means[col] = df[col].mean()
            std_val = df[col].std()
            self.stds[col] = std_val if std_val != 0 else 1e-8

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df_scaled = df.copy()
        for col in cols:
            df_scaled[col] = (df_scaled[col] - self.means[col]) / self.stds[col]
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        self.fit(df, cols)
        return self.transform(df, cols)


class MinMaxScaler:
 
    def __init__(self) -> None:
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> None:
        for col in cols:
            self.mins[col] = df[col].min()
            self.maxs[col] = df[col].max()

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df_scaled = df.copy()
        for col in cols:
            mn = self.mins[col]
            mx = self.maxs[col]
            denom = (mx - mn) if (mx - mn) != 0 else 1e-8
            df_scaled[col] = (df_scaled[col] - mn) / denom
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        self.fit(df, cols)
        return self.transform(df, cols)

class DenseLayer:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        limit = 1.0 / math.sqrt(input_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        self.dW = (x.T @ grad_output) / batch_size
        self.db = grad_output.mean(axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input

class ReLU:
    def __init__(self) -> None:
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output.copy()
        grad[self.x < 0] = 0
        return grad


class Sigmoid:
    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.out * (1 - self.out))


def bce_loss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred_clamped = np.clip(pred, eps, 1 - eps)
    return -np.mean(
        target * np.log(pred_clamped) + (1 - target) * np.log(1 - pred_clamped)
    )


def bce_loss_grad(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    pred_clamped = np.clip(pred, eps, 1 - eps)
    return (pred_clamped - target) / (pred_clamped * (1 - pred_clamped) + eps)


class NeuralNetwork:

    def __init__(self, input_dim: int) -> None:
        self.layer1 = DenseLayer(input_dim, 32)
        self.act1 = ReLU()
        self.layer2 = DenseLayer(32, 64)
        self.act2 = ReLU()
        self.layer3 = DenseLayer(64, 32)
        self.act3 = ReLU()
        self.layer4 = DenseLayer(32, 1)
        self.act4 = Sigmoid()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out1 = self.layer1.forward(x)
        out1a = self.act1.forward(out1)
        out2 = self.layer2.forward(out1a)
        out2a = self.act2.forward(out2)
        out3 = self.layer3.forward(out2a)
        out3a = self.act3.forward(out3)
        out4 = self.layer4.forward(out3a)
        out4a = self.act4.forward(out4)
        return out4a

    def forward_and_backward(self, x: np.ndarray, y: np.ndarray) -> float:
        out1 = self.layer1.forward(x)
        out1a = self.act1.forward(out1)
        out2 = self.layer2.forward(out1a)
        out2a = self.act2.forward(out2)
        out3 = self.layer3.forward(out2a)
        out3a = self.act3.forward(out3)
        out4 = self.layer4.forward(out3a)
        pred = self.act4.forward(out4)

        loss_val = bce_loss(pred, y)
        grad_pred = bce_loss_grad(pred, y)

        grad_out4 = self.act4.backward(grad_pred)
        grad_out3a = self.layer4.backward(out3a, grad_out4)
        grad_out3 = self.act3.backward(grad_out3a)
        grad_out2a = self.layer3.backward(out2a, grad_out3)
        grad_out2 = self.act2.backward(grad_out2a)
        grad_out1a = self.layer2.backward(out1a, grad_out2)
        _ = self.act1.backward(grad_out1a)
        _ = self.layer1.backward(x, grad_out1a)

        return loss_val

    def parameters(self) -> List[DenseLayer]:
        return [self.layer1, self.layer2, self.layer3, self.layer4]


class AdamOptimizer:
    def __init__(self, params: List[DenseLayer], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for layer in self.params:
            if layer.dW is None or layer.db is None:
                continue
            layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db

            layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)

            mW_hat = layer.mW / (1 - self.beta1 ** self.t)
            mb_hat = layer.mb / (1 - self.beta1 ** self.t)
            vW_hat = layer.vW / (1 - self.beta2 ** self.t)
            vb_hat = layer.vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps))
            layer.b -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))


def compute_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred_binary = (pred >= threshold).astype(int)
    tp = np.sum((pred_binary == 1) & (target == 1))
    tn = np.sum((pred_binary == 0) & (target == 0))
    fp = np.sum((pred_binary == 1) & (target == 0))
    fn = np.sum((pred_binary == 0) & (target == 1))

    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = numerator / denominator

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(mcc),
    }


def train_model(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    optimizer = AdamOptimizer(model.parameters(), lr=lr)
    n_samples = X_train.shape[0]
    n_batches = int(math.ceil(n_samples / batch_size))

    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            loss = model.forward_and_backward(x_batch, y_batch)
            optimizer.step()
            epoch_loss += loss

        avg_loss = epoch_loss / n_batches

        val_preds = model.forward(X_val)
        val_loss = bce_loss(val_preds, y_val)
        metrics = compute_metrics(val_preds, y_val)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"Prec: {metrics['precision']:.3f} | "
            f"Rec: {metrics['recall']:.3f} | "
            f"MCC: {metrics['mcc']:.3f}"
        )
