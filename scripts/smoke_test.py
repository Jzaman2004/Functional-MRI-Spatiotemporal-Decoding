import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
    def forward(self, x):
        return self.linear(x)

class PyTorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, n_classes, device, epochs=5, lr=0.01, batch_size=16):
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = LinearClassifier(self.n_features, self.n_classes).to(self.device)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                out = self.model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
        return self
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            preds = torch.argmax(out, dim=1).cpu().numpy()
        return preds
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

if __name__ == '__main__':
    # Small synthetic dataset
    n_samples = 200
    n_features = 50
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create a simple linear separable label
    true_w = np.random.randn(n_features)
    logits = X.dot(true_w)
    y = (logits > np.median(logits)).astype(int)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    clf = PyTorchSklearnWrapper(n_features=n_features, n_classes=2, device=device, epochs=10)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f'Smoke test accuracy on synthetic data: {acc*100:.2f}%')
