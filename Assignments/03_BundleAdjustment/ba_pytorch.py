import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def euler_xyz_to_matrix(angles):
    """Convert XYZ Euler angles in radians to rotation matrices."""
    x, y, z = angles.unbind(dim=-1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    one = torch.ones_like(x)
    zero = torch.zeros_like(x)

    rx = torch.stack(
        [
            one,
            zero,
            zero,
            zero,
            cx,
            -sx,
            zero,
            sx,
            cx,
        ],
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)
    ry = torch.stack(
        [
            cy,
            zero,
            sy,
            zero,
            one,
            zero,
            -sy,
            zero,
            cy,
        ],
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)
    rz = torch.stack(
        [
            cz,
            -sz,
            zero,
            sz,
            cz,
            zero,
            zero,
            zero,
            one,
        ],
        dim=-1,
    ).reshape(*angles.shape[:-1], 3, 3)
    return rx @ ry @ rz


def load_observations(data_dir):
    points2d = np.load(data_dir / "points2d.npz")
    keys = sorted(points2d.files)
    obs = np.stack([points2d[key] for key in keys], axis=0).astype(np.float32)
    visibility = obs[..., 2] > 0.5
    view_ids, point_ids = np.nonzero(visibility)
    xy = obs[view_ids, point_ids, :2]
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)
    return keys, obs, view_ids.astype(np.int64), point_ids.astype(np.int64), xy, colors


def initial_points(obs, f0, distance):
    xy = obs[..., :2]
    vis = obs[..., 2:3] > 0.5
    valid = np.maximum(vis.sum(axis=0), 1.0)
    x0 = ((xy[..., 0:1] - CX) * distance / f0 * vis).sum(axis=0) / valid
    y0 = (-(xy[..., 1:2] - CY) * distance / f0 * vis).sum(axis=0) / valid
    rng = np.random.default_rng(2026)
    z0 = rng.normal(loc=0.0, scale=0.08, size=x0.shape).astype(np.float32)
    pts = np.concatenate([x0, y0, z0], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(pts[:, :2], axis=1), 95)
    if scale > 1e-6:
        pts /= scale
    return pts


def project(points, euler, translation, log_f, view_ids, point_ids):
    rotations = euler_xyz_to_matrix(euler[view_ids])
    pts = points[point_ids]
    cam = torch.bmm(rotations, pts.unsqueeze(-1)).squeeze(-1) + translation[view_ids]
    z = cam[:, 2].clamp(max=-1e-4)
    f = torch.exp(log_f)
    u = -f * cam[:, 0] / z + CX
    v = f * cam[:, 1] / z + CY
    return torch.stack([u, v], dim=-1), cam


def reprojection_loss(pred_xy, target_xy):
    residual = pred_xy - target_xy
    # Charbonnier loss keeps a few difficult boundary points from dominating.
    return torch.sqrt((residual * residual).sum(dim=-1) + 1e-3).mean()


def save_obj(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(
                "v {:.8f} {:.8f} {:.8f} {:.6f} {:.6f} {:.6f}\n".format(
                    p[0], p[1], p[2], c[0], c[1], c[2]
                )
            )


def save_loss_plot(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean reprojection error (px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_point_cloud_preview(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(points) // 12000)
    pts = points[::step]
    cols = colors[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1.0, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=8, azim=-85)
    ax.set_box_aspect(
        (
            np.ptp(points[:, 0]) + 1e-6,
            np.ptp(points[:, 1]) + 1e-6,
            np.ptp(points[:, 2]) + 1e-6,
        )
    )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def optimize(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, obs, view_ids_np, point_ids_np, xy_np, colors = load_observations(data_dir)
    n_views, n_points = obs.shape[:2]
    print(f"Loaded {n_views} views, {n_points} points, {len(xy_np)} visible observations")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fov_rad = math.radians(args.init_fov)
    f0 = IMAGE_SIZE / (2.0 * math.tan(fov_rad / 2.0))
    points0 = initial_points(obs, f0=f0, distance=args.init_distance)

    yaw = np.linspace(math.radians(args.init_yaw_deg), -math.radians(args.init_yaw_deg), n_views)
    euler0 = np.zeros((n_views, 3), dtype=np.float32)
    euler0[:, 1] = yaw.astype(np.float32)
    translation0 = np.zeros((n_views, 3), dtype=np.float32)
    translation0[:, 2] = -args.init_distance

    points = torch.nn.Parameter(torch.from_numpy(points0).to(device))
    euler = torch.nn.Parameter(torch.from_numpy(euler0).to(device))
    translation = torch.nn.Parameter(torch.from_numpy(translation0).to(device))
    log_f = torch.nn.Parameter(torch.tensor(math.log(f0), dtype=torch.float32, device=device))

    view_ids = torch.from_numpy(view_ids_np).to(device)
    point_ids = torch.from_numpy(point_ids_np).to(device)
    target_xy = torch.from_numpy(xy_np).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": args.lr_points},
            {"params": [euler], "lr": args.lr_cameras},
            {"params": [translation], "lr": args.lr_cameras},
            {"params": [log_f], "lr": args.lr_focal},
        ]
    )

    history = []
    n_obs = target_xy.shape[0]
    batch_size = min(args.batch_size, n_obs)
    for it in range(1, args.iters + 1):
        batch = torch.randint(0, n_obs, (batch_size,), device=device)
        pred_xy, cam = project(points, euler, translation, log_f, view_ids[batch], point_ids[batch])
        data_loss = reprojection_loss(pred_xy, target_xy[batch])

        center_reg = points.mean(dim=0).pow(2).sum()
        z_reg = torch.relu(cam[:, 2] + 1e-3).pow(2).mean()
        focal_reg = (log_f - math.log(f0)).pow(2)
        loss = data_loss + args.center_reg * center_reg + args.depth_reg * z_reg + args.focal_reg * focal_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        err = float(data_loss.detach().cpu())
        history.append(err)
        if it == 1 or it % args.log_every == 0:
            print(f"iter {it:05d}  reproj={err:.4f}px  f={float(torch.exp(log_f).detach().cpu()):.2f}")

    with torch.no_grad():
        if args.eval_batch_size <= 0:
            eval_batch = n_obs
        else:
            eval_batch = args.eval_batch_size
        errors = []
        for start in range(0, n_obs, eval_batch):
            end = min(start + eval_batch, n_obs)
            pred_xy, _ = project(points, euler, translation, log_f, view_ids[start:end], point_ids[start:end])
            errors.append(torch.linalg.norm(pred_xy - target_xy[start:end], dim=-1).detach().cpu())
        all_errors = torch.cat(errors).numpy()

    points_np = points.detach().cpu().numpy()
    euler_np = euler.detach().cpu().numpy()
    translation_np = translation.detach().cpu().numpy()
    focal = float(torch.exp(log_f).detach().cpu())

    save_obj(out_dir / "reconstruction.obj", points_np, colors)
    save_loss_plot(out_dir / "loss_curve.png", history)
    save_point_cloud_preview(out_dir / "point_cloud_preview.png", points_np, colors)
    np.savez(
        out_dir / "ba_parameters.npz",
        points3d=points_np,
        euler_xyz=euler_np,
        translation=translation_np,
        focal=np.array([focal], dtype=np.float32),
        history=np.array(history, dtype=np.float32),
        view_keys=np.array(keys),
    )
    summary = {
        "views": int(n_views),
        "points": int(n_points),
        "visible_observations": int(n_obs),
        "device": str(device),
        "iterations": int(args.iters),
        "focal": focal,
        "mean_reprojection_error_px": float(all_errors.mean()),
        "median_reprojection_error_px": float(np.median(all_errors)),
        "p90_reprojection_error_px": float(np.percentile(all_errors, 90)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved OBJ, parameters, loss curve, and summary to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch bundle adjustment for Assignment 3.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/ba")
    parser.add_argument("--device", default=None, help="cuda, cpu, or empty for auto")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--eval-batch-size", type=int, default=262144)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-fov", type=float, default=55.0)
    parser.add_argument("--init-distance", type=float, default=2.8)
    parser.add_argument("--init-yaw-deg", type=float, default=70.0)
    parser.add_argument("--lr-points", type=float, default=0.01)
    parser.add_argument("--lr-cameras", type=float, default=0.002)
    parser.add_argument("--lr-focal", type=float, default=0.001)
    parser.add_argument("--center-reg", type=float, default=1e-3)
    parser.add_argument("--depth-reg", type=float, default=10.0)
    parser.add_argument("--focal-reg", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    optimize(parse_args())
