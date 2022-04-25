import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from cnntdanet.tda import get_tda_pipeline
from cnntdanet.utils import load_fits


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='./datasets/20220209_LOAO_check/bogus')
    parser.add_argument('--dir_model', type=str, default='./outputs/gradcam/gradcamoncnn-bc.h5')
    parser.add_argument('--method', type=str, default='betti-curve')

    # For generating tda features
    parser.add_argument('--n_jobs',   type=int, default=None)
    parser.add_argument('--n_bins',   type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)

    args = parser.parse_args()

    # Exception
    if (args.method in ['persistence-image', 'betti-curve']) and (args.n_bins is None): 
        parser.error(f"Argument [--n_bins] is required for [--method]='{args.method}'")

    elif (args.method in ['persistence-landscape']) and ((args.n_bins is None) or (args.n_layers is None)): 
        parser.error(f"Argument [--n_bins] and [--n_layers] are required for [--method]='{args.method}'")
    
    return args


def prepare_dataset(dir_data, pipeline):
    if not os.path.exists(dir_data):
        raise ValueError(f"{dir_data} does not exist.")
    img_list = glob(f'{dir_data}/*.fits')
    id = list(map(lambda x: x.split('/')[-1], img_list))

    # Loading and preprocessing dataset
    X_img = list(map(lambda x: load_fits(x), img_list))
    X_img = np.array(X_img)
    X_tda = pipeline.fit_transform(X_img)
    
    # Create a pd.DataFrame to store predictions
    df = pd.DataFrame({'id': id})
    df['pred'] = np.nan
    df['prob'] = np.nan

    return X_img, X_tda, df
    
    
def main():
    args = get_argument()

    model = tf.keras.models.load_model(args.dir_model)
    pipeline = get_tda_pipeline(method=args.method, n_jobs=args.n_jobs, n_bins=args.n_bins, n_layers=args.n_layers)
    X_img, X_tda, df = prepare_dataset(args.dir_data, pipeline)

    for i, (img, tda) in tqdm(enumerate(zip(X_img, X_tda)), total=len(X_img)):
        img = img[np.newaxis]
        tda = tda[np.newaxis]

        pred = model.predict([img, tda])
        prob = np.max(pred, axis=1)[0]
        pred = np.argmax(pred, axis=1)[0]

        df.loc[i, 'pred'] = pred
        df.loc[i, 'prob'] = prob

    df.to_csv('./result.csv', index=False)


if __name__ == '__main__':
    main()
