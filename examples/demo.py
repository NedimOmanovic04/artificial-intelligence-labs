"""Reproducible training/evaluation on scikit-learn's bundled digits dataset."""
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def train_experiment():
    """Choose regularization using training-fold CV; evaluate on held-out data."""
    data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=SEED
    )
    pipeline = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(32,), max_iter=200, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=15, random_state=SEED,
    ))
    search = GridSearchCV(pipeline, {'mlpclassifier__alpha': [0.001, 0.01]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                          scoring='f1_macro', n_jobs=1, error_score='raise')
    search.fit(x_train, y_train)
    prediction = search.predict(x_test)
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    report = {
        'dataset': 'sklearn.datasets.load_digits (bundled, 8x8 images)',
        'seed': SEED, 'train_samples': len(y_train), 'test_samples': len(y_test),
        'selection': '3-fold stratified CV on training data only; macro F1',
        'best_params': search.best_params_, 'best_cv_macro_f1': search.best_score_,
        'test_accuracy': accuracy_score(y_test, prediction),
        'test_macro_f1': f1_score(y_test, prediction, average='macro'),
        'dummy_test_accuracy': accuracy_score(y_test, dummy.predict(x_test)),
        'classification_report': classification_report(y_test, prediction, output_dict=True),
        'cv_candidates': [
            {'params': params, 'mean_macro_f1': float(mean), 'std_macro_f1': float(std)}
            for params, mean, std in zip(search.cv_results_['params'],
                                        search.cv_results_['mean_test_score'],
                                        search.cv_results_['std_test_score'])
        ],
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scikit-learn': sklearn.__version__},
    }
    return report, search.best_estimator_, y_test, prediction


def main():
    output = Path(__file__).resolve().parents[1] / 'reports'
    output.mkdir(exist_ok=True)
    report, model, y_test, prediction = train_experiment()
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, cmap='Blues')
    plt.title('Digits MLP - held-out test set')
    plt.tight_layout()
    plt.savefig(output / 'confusion-matrix.png', dpi=140)
    plt.close()
    network = model.named_steps['mlpclassifier']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(network.loss_curve_)
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Cross entropy + penalty')
    axes[1].plot(network.validation_scores_)
    axes[1].set(title='Internal validation accuracy', xlabel='Epoch', ylabel='Accuracy')
    fig.tight_layout()
    fig.savefig(output / 'training-curves.png', dpi=140)
    plt.close(fig)
    print(json.dumps({key: report[key] for key in (
        'best_cv_macro_f1', 'test_accuracy', 'test_macro_f1', 'dummy_test_accuracy')}, indent=2))


if __name__ == '__main__':
    main()
