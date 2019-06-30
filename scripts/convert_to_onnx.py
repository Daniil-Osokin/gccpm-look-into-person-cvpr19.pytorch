import argparse

import torch

from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name):
    input_names = ['data']
    input = torch.randn(1, 3, 256, 256)
    output_names = ['stage_1_output_1_heatmaps',
                    'stage_2_output_1_heatmaps']

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    net = SinglePersonPoseEstimationWithMobileNet(to_onnx=True)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name)
