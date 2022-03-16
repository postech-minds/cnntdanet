import os
import tensorflow as tf

from argparse import ArgumentParser 
from cnntdanet.models import get_cnn_net, get_cnn_tda_net
from cnntdanet.utils import seed_all
from cnntdanet.utils.datasets import prepare_dataset
from cnntdanet.utils.plotting import plot_learning_curve


def get_argument():
    parser = ArgumentParser()
    # Experiment setting
    parser.add_argument('--seed',     type=int, default=0)
    parser.add_argument('--dataset',  type=str)
    parser.add_argument('--method',   type=str, default=None)
    parser.add_argument('--dir_save', type=str, default='./outputs')

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
    return args


def main():
    args = get_argument()
    seed_all(args.seed)

    if not os.path.exists(args.dir_save):
        os.makedirs(f'{args.dir_save}')

    dataset = prepare_dataset(dataset=args.dataset, method=args.method, n_jobs=args.n_jobs, n_bins=args.n_bins, n_layers=args.n_layers)
    
    if args.method is not None:
        X_img, X_tda, y = dataset['X_img'], dataset['X_tda'], dataset['y']
        n_classes = y.shape[-1]
        input_shape = {'local': X_img.shape[1:], 'global': X_tda.shape[1:]}
        model = get_cnn_tda_net(method=args.method, input_shape=input_shape, n_classes=n_classes)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=tf.keras.optimizers.Adam(lr=args.lr))
        history = model.fit(
            [X_img, X_tda], y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

    elif args.method is None:
        X_img, y = dataset['X_img'], dataset['y']
        model = get_cnn_net(input_shape=X_img.shape[1:], n_classes=y.shape[-1])
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer='adam')
        history = model.fit(
            X_img, y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

    model.save(f'{args.dir_save}/model.h5')
    plot_learning_curve(history.history, args.dir_save)


if __name__ == '__main__':
    main()
