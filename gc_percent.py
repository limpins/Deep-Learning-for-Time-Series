"""
Email: autuanliu@163.com
Date: 2018/10/15
"""

from core import plot_save_gc_precent


def main():
    for signal_type in ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals']:
        plot_save_gc_precent(f'checkpoints/{signal_type}_granger_matrix.txt',
                             f'images/{signal_type}_granger_matrix%.png', f'{signal_type}_granger_matrix%', f'checkpoints/{signal_type}_granger_matrix%.txt')


if __name__ == '__main__':
    main()
