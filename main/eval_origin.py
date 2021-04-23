"""
计算跟踪结果与 real GT 的精度
"""
import sys
sys.path.append('/home/etvuz/projects/adversarial_attack/video_analyst')
import os
import argparse
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k


def main():
    experiment = ExperimentGOT10k('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/GOT-10k', subset=subset,
                                  result_dir=result_dir, report_dir=report_dir)
    experiment.report([network])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--loop_num', type=int)
    parsed_args = parser.parse_args()

    subset='val'
    network = 'siamfcpp_googlenet'
    iter_num = parsed_args.loop_num
    result_dir= '/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/result/siamfcpp_googlenet/{}'.format(iter_num)
    report_dir='/home/etvuz/projects/adversarial_attack/video_analyst/snapshots_imperceptible_patch/64/report/siamfcpp_googlenet/{}'.format(iter_num)
    main()