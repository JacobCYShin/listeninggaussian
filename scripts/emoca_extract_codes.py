import argparse
import warnings
from pathlib import Path

import torch
import numpy as np

try:
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    torch.serialization.add_safe_globals([ModelCheckpoint])
except Exception:
    pass

from gdl_apps.EMOCA.utils.load import load_model
from gdl_apps.EMOCA.utils.io import save_codes, save_images, save_obj
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM


def parse_args():
    parser = argparse.ArgumentParser(description="Extract EMOCA codes from a video without rendering a reconstruction.")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path.")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for EMOCA codes.")
    parser.add_argument("--model_name", type=str, default="EMOCA_v2_lr_mse_20")
    parser.add_argument("--path_to_models", type=str, required=True)
    parser.add_argument("--mode", type=str, default="detail", choices=["detail", "coarse"])
    parser.add_argument("--save_images", action="store_true", help="Also save debug images.")
    parser.add_argument("--save_mesh", action="store_true", help="Also save meshes.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for EMOCA inference.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for EMOCA inference.")
    parser.add_argument("--flame_out_dir", type=str, default="", help="Optional output dir for flame_params.npy.")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    warnings.filterwarnings(
        "ignore",
        message="Default grid_sample and affine_grid behavior has changed to align_corners=False",
        category=UserWarning,
    )

    device = torch.device(args.device)
    dm = TestFaceVideoDM(
        args.input_video,
        args.output_folder,
        batch_size=args.batch_size,
        num_workers=0,
        device=device,
    )
    dm.prepare_data()
    dm.setup()

    emoca, _ = load_model(args.path_to_models, args.model_name, args.mode)
    emoca = emoca.to(device)
    emoca.eval()

    if Path(args.output_folder).is_absolute():
        outfolder = Path(args.output_folder)
    else:
        outfolder = Path(args.output_folder).resolve()

    dl = dm.test_dataloader()
    for batch in dl:
        current_bs = batch["image"].shape[0]
        with torch.no_grad():
            batch["image"] = batch["image"].to(device)
            if len(batch["image"].shape) == 3:
                batch["image"] = batch["image"].view(1, 3, 224, 224)
            vals = emoca.encode(batch, training=False)
            decoded = emoca.decode(vals, training=False)
            if isinstance(decoded, tuple):
                vals = decoded[0]
                visdict = decoded[1] if len(decoded) > 1 else {}
            else:
                vals = decoded
                visdict = {}
        for i in range(current_bs):
            name = batch["image_name"][i]
            sample_output_folder = outfolder / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)
            if args.save_mesh:
                save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
            if args.save_images:
                save_images(outfolder, name, visdict, i)
            save_codes(outfolder, name, vals, i)

    # Build flame_params.npy from exp/pose if possible.
    flame_dirs = sorted([p for p in outfolder.iterdir() if p.is_dir() and (p / "exp.npy").exists() and (p / "pose.npy").exists()])
    if flame_dirs:
        flame_params = []
        for p in flame_dirs:
            exp = np.load(p / "exp.npy").reshape(-1)[:50]
            pose = np.load(p / "pose.npy").reshape(-1)[:6]
            flame_params.append(np.concatenate([exp, pose]))
        arr = np.stack(flame_params, axis=0)

        flame_out_dir = args.flame_out_dir
        if not flame_out_dir:
            base = outfolder.name
            if base.startswith("emoca_"):
                guess = Path("/workspace/data") / base.replace("emoca_", "", 1)
                if guess.exists():
                    flame_out_dir = str(guess)
        if not flame_out_dir:
            flame_out_dir = str(outfolder)
        flame_out_path = Path(flame_out_dir) / "flame_params.npy"
        flame_stats_path = Path(flame_out_dir) / "flame_params_stats.npz"
        flame_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(flame_out_path, arr)
        np.savez(flame_stats_path, mean=arr.mean(axis=0), std=arr.std(axis=0))
        print(f"[FLAME] Saved {flame_out_path} shape={arr.shape}")

    print(f"Done. Codes saved to: {outfolder}")


if __name__ == "__main__":
    main()
