"""
Email: autuanliu@163.com
Date: 2018/10/15
"""

from core import plot_save_gc_precent


def main():
    for policy in ['with_NUE', 'without_NUE']:
        for signal_type in ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals']:
            in1 = f'checkpoints/{policy}/{signal_type}_granger_matrix.txt'
            in2 = f'images/{policy}/{signal_type}_granger_matrix%.png'
            in3 = f'{signal_type}_granger_matrix%'
            in4 = f'checkpoints/{policy}/{signal_type}_granger_matrix%.txt'
            plot_save_gc_precent(in1, in2, in3, in4)


if __name__ == '__main__':
    main()
