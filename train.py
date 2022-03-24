import os
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from cnntdanet.models import get_cnn_net, get_cnn_tda_net
from cnntdanet.utils import seed_all
from cnntdanet.utils.datasets import prepare_dataset
from cnntdanet.utils.plotting import plot_learning_curve


def get_argument():
    parser = ArgumentParser()
    # Experiment setting
    parser.add_argument('--seed',      type=int,   default=0)
    parser.add_argument('--dataset',   type=str)
    parser.add_argument('--dir_data',  type=str)
    parser.add_argument('--method',    type=str,   default=None)
    parser.add_argument('--dir_save',  type=str,   default='./outputs')
    parser.add_argument('--test_size', type=float, default=0.0)

    # For generating tda features
    parser.add_argument('--n_jobs',   type=int, default=-1)
    parser.add_argument('--n_bins',   type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)

    # Training Configuration
    parser.add_argument('--lr',               type=float, default=0.001)
    parser.add_argument('--epochs',           type=int,   default=1)
    parser.add_argument('--batch_size',       type=int,   default=32)
    parser.add_argument('--validation_split', type=float, default=0.3)
    
    args = parser.parse_args()

    # Exception
    if (args.dataset in ['skin-cancer', 'transient-vs-bogus']) and (args.dir_data is None):
        parser.error(f"Argument [--dir_data] is required for [--dataset]='{args.dataset}'")

    elif (args.method in ['persistence-image', 'betti-curve']) and (args.n_bins is None): 
        parser.error(f"Argument [--n_bins] is required for [--method]='{args.method}'")

    elif (args.method in ['persistence-landscape']) and ((args.n_bins is None) or (args.n_layers is None)): 
        parser.error(f"Argument [--n_bins] and [--n_layers] are required for [--method]='{args.method}'")
    
    return args


def main():
    args = get_argument()
    seed_all(args.seed)

    if not os.path.exists(args.dir_save):
        os.makedirs(f'{args.dir_save}')

    dataset = prepare_dataset(dataset=args.dataset, dir_data=args.dir_data, method=args.method, n_jobs=args.n_jobs, n_bins=args.n_bins, n_layers=args.n_layers)
    
    if args.method is not None:
        X_img, X_tda, y = dataset['X_img'], dataset['X_tda'], dataset['y']
        print(X_img.shape, X_tda.shape, y.shape)
        if args.test_size > 0:
            y_ = np.argmax(y, axis=1)
            train_indices, test_indices = train_test_split(np.arange(len(X_img)), test_size=args.test_size, shuffle=True, random_state=args.seed, stratify=y_)
            X_img_test, X_tda_test, y_test = X_img[test_indices], X_tda[test_indices],  y[test_indices]
            X_img, X_tda, y = X_img[train_indices], X_tda[train_indices], y[train_indices]
        n_classes = y.shape[-1]
        input_shape = {'local': X_img.shape[1:], 'global': X_tda.shape[1:]}

        # Training
        model = get_cnn_tda_net(method=args.method, input_shape=input_shape, n_classes=n_classes)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=tf.keras.optimizers.Adam(lr=args.lr))
        history = model.fit(
            [X_img, X_tda], y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        # Evaluation
        if args.test_size > 0:
            y_pred = model.predict([X_img_test, X_tda_test])
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            print(f"Test accuracy: {100 * accuracy_score(y_true, y_pred):.2f}")

    elif args.method is None:
        X_img, y = dataset['X_img'], dataset['y']
        if args.test_size > 0:
            y_ = np.argmax(y, axis=1)
            train_indices, test_indices = train_test_split(np.arange(len(X_img)), test_size=args.test_size, shuffle=True, random_state=args.seed, stratify=y_)
            X_img_test, y_test = X_img[test_indices], y[test_indices]
            X_img, y = X_img[train_indices], y[train_indices]

        # Training
        model = get_cnn_net(input_shape=X_img.shape[1:], n_classes=y.shape[-1])
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=tf.keras.optimizers.Adam(lr=args.lr))
        history = model.fit(
            X_img, y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )
        # Evaluation
        if args.test_size > 0:
            y_pred = model.predict(X_img_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            print(f"Test accuracy: {100 * accuracy_score(y_true, y_pred):.2f}")

    model.save(f'{args.dir_save}/model.h5')
    plot_learning_curve(history.history, args.dir_save)


if __name__ == '__main__':
    main()
