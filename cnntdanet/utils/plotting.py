import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def plot_learning_curve(history, dir_save):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    e = np.arange(len(history['loss']), dtype=np.int64)

    fig.suptitle("Learning curves", fontsize=18)
    axes[0].plot(e, history['loss'], 'bo-', label='train')
    axes[0].plot(e, history['val_loss'], 'ro-', label='valid')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(e, history['acc'], 'bo-', label='train')
    axes[1].plot(e, history['val_acc'], 'ro-', label='valid')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid()

    plt.savefig(f'{dir_save}/learning-curve.pdf', dpi=250)
