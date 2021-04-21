"""
计算跟踪结果与 real GT 的精度
"""

import os

from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k


def main():
    experiment = ExperimentGOT10k(os.path.join(root, 'video_analyst/datasets/GOT-10k'), subset='val',
                                  result_dir='/tmp/result', report_dir='/tmp/result/report')
    experiment.report(['siamfcpp_alexnet'])


if __name__ == '__main__':
    root = '/home/etvuz/projects/adversarial_attack'
    main()