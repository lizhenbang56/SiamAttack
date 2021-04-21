"""
计算跟踪结果与 real GT 的精度
"""
import sys
sys.path.append('/home/etvuz/projects/adversarial_attack/video_analyst')
import os

from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k


def main():
    experiment = ExperimentGOT10k('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/GOT-10k', subset=subset,
                                  result_dir=result_dir, report_dir=report_dir)
    experiment.report([network])


if __name__ == '__main__':
    subset='val'
    network = 'siamfcpp_googlenet'
    result_dir='/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_small_patch/32/result/GOT-10k/siamfcpp_googlenet/2048'
    report_dir='/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_small_patch/32/report/GOT-10k/2048'
    main()