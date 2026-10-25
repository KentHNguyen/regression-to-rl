"""
Part 2: Supervised learning for classification (MLP on Iris)
"""
import warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def report(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train) * 100
    test_acc = model.score(X_test, y_test) * 100
    print(f"{name}")
    print(f"  Training accuracy: {train_acc:.2f}%")
    print(f"  Testing accuracy:  {test_acc:.2f}%\n")
    return model


# ---------- (1) Load Iris ----------
iris = load_iris()
X, y = iris.data, iris.target

# ---------- (2) Split 80% train / 20% test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- (3) Baseline: 1 hidden layer, 3 neurons ----------
baseline = MLPClassifier(hidden_layer_sizes=(3,), max_iter=200, random_state=1)
report("(3) Baseline MLP: 1 layer x 3 neurons", baseline,
       X_train, y_train, X_test, y_test)

# ---------- (4) Tune hyperparameters (still 1 layer x 3 neurons) ----------
# Changes: scale the inputs, train longer, use tanh and a higher learning rate.
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

tuned = MLPClassifier(hidden_layer_sizes=(3,), activation="tanh",
                      solver="adam", learning_rate_init=0.01,
                      max_iter=2000, random_state=1)
report("(4) Tuned MLP: 1 layer x 3 neurons (scaled, tanh, lr=0.01, 2000 iters)",
       tuned, X_train_s, y_train, X_test_s, y_test)

# ---------- (5) Bigger network: more layers and neurons ----------
bigger = MLPClassifier(hidden_layer_sizes=(20, 20), activation="relu",
                       solver="adam", learning_rate_init=0.01,
                       max_iter=2000, random_state=1)
report("(5) Bigger MLP: 2 layers x 20 neurons", bigger,
       X_train_s, y_train, X_test_s, y_test)
