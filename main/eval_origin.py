"""
计算跟踪结果与 real GT 的精度
"""
import sys

from cv2 import phase
from paths import ROOT_PATH
import os
import argparse
from videoanalyst.evaluation.got_benchmark.experiments.got10k import ExperimentGOT10k
from videoanalyst.evaluation.got_benchmark.experiments.otb import ExperimentOTB
from videoanalyst.evaluation.got_benchmark.experiments.lasot import ExperimentLaSOT


def main():
    if parsed_args.dataset_name == 'GOT-10k_Val':
        experiment = ExperimentGOT10k(os.path.join(sys.path[0], 'datasets/GOT-10k'), subset='val',
                                      result_dir=result_dir, report_dir=report_dir, phase='eval')
        experiment.report(['siamfcpp_googlenet'])
    elif parsed_args.dataset_name == 'LaSOT':
        experiment = ExperimentLaSOT('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/LaSOT',
                                 subset='test',
                                 return_meta=False,
                                 result_dir=result_dir, report_dir=report_dir)
        experiment.report(['siamfcpp_googlenet'])
    elif parsed_args.dataset_name == 'OTB_2015':
        experiment = ExperimentOTB('/home/etvuz/projects/adversarial_attack/video_analyst/datasets/OTB/OTB2015',
                               version=2015,
                               result_dir=result_dir, report_dir=report_dir)
        experiment.report(['siamfcpp_googlenet'])
    else:
        assert False, parsed_args


if __name__ == '__main__':
    
    """定义参数"""
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--loop_num', type=int, default=1024)
    parser.add_argument('--dataset_name', type=str, default='GOT-10k_Val')  #  'LaSOT' 'GOT-10k_Val' 'OTB_2015'
    parsed_args = parser.parse_args()
    """定义参数"""

    """定义文件夹"""
    result_dir = os.path.join(sys.path[0], 'snapshots_imperceptible_patch/64/result/{}/siamfcpp_googlenet/{}'.format(parsed_args.dataset_name, parsed_args.loop_num))
    report_dir = os.path.join(sys.path[0], 'snapshots_imperceptible_patch/64/report/{}/siamfcpp_googlenet/{}'.format(parsed_args.dataset_name, parsed_args.loop_num))
    """定义文件夹"""

    """运行主程序"""
    main()
    """运行主程序"""