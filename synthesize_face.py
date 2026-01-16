#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_motion
from utils.general_utils import safe_state
from utils.camera_utils import loadCamOnTheFly
import copy
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, MotionNetwork


def render_set(model_path, name, iteration, views, gaussians, motion_net, pipeline, background, start_idx, num_frames):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    all_preds = []
    all_gts = []

    if start_idx < 0:
        start_idx = 0
    if num_frames > 0:
        end_idx = min(start_idx + num_frames, len(views))
        views = views[start_idx:end_idx]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True)):
        if view.original_image == None:
            view = loadCamOnTheFly(copy.deepcopy(view))
        with torch.no_grad():
            render_pkg = render_motion(view, gaussians, motion_net, pipeline, background, frame_idx=0)

        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        alpha = render_pkg["alpha"]
        image = image - background[:, None, None] * (1.0 - alpha) + view.background.cuda() / 255.0 * (1.0 - alpha)
        gt_image = torch.clamp(view.original_image.to("cuda") / 255.0, 0.0, 1.0)

        pred = (image[0:3, ...].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        all_preds.append(pred)
        all_gts.append((gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    imageio.mimwrite(os.path.join(render_path, "out.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)
    imageio.mimwrite(os.path.join(gts_path, "out.mp4"), all_gts, fps=25, quality=8, macro_block_size=1)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, use_train: bool, start_idx: int, num_frames: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        motion_net = MotionNetwork(args=dataset).cuda()
        ckpt_path = os.path.join(dataset.model_path, "chkpnt_face_latest.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        (model_params, motion_params, _, _) = torch.load(ckpt_path)
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)

        bg_color = [0, 1, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(
            dataset.model_path,
            "train" if use_train else "test",
            scene.loaded_iter if scene.loaded_iter else "face_latest",
            scene.getTrainCameras() if use_train else scene.getTestCameras(),
            gaussians,
            motion_net,
            pipeline,
            background,
            start_idx,
            num_frames,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Face-only rendering script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=0)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.use_train, args.start_idx, args.num_frames)
