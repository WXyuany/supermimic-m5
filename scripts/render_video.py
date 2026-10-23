"""Render a deterministic policy beside its reference; does not alter training."""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')
import argparse, json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--clip', default='walk1_subject1')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu' if sys.platform == 'darwin' else 'cuda:0')
    parser.add_argument('--npz-dir', type=Path, default=Path('data/npz/lafan1'))
    args = parser.parse_args()
    policy, config = load_policy_from_checkpoint(args.ckpt, args.device)
    motion = args.npz_dir / (args.clip + '.npz')
    assert motion.is_file(), motion
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(motion)], num_envs=1,
        future_frames=config['env']['future_frames'], episode_length_s=1e5,
        action_noise_std=0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode='start', record_transitions=False,
        anchor_obs=config['env'].get('anchor_obs', False), seed=0), device=args.device)
    renderer = writer = None
    frames, failed, steps = 0, False, 0
    try:
        cmd = env.command_manager.get_term('motion')
        cmd.forced_frame[:] = cmd.library.clip_start[0]
        obs, _ = env.reset()
        model = env.sim.mj_model
        assert model.nq == 36
        model.vis.global_.offwidth = 640
        model.vis.global_.offheight = 480
        actual, reference = mujoco.MjData(model), mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = 3.2, 135, -15
        font_path = next((p for p in (
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
        ) if Path(p).exists()), None)
        font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=25, codec='libx264', quality=7, macro_block_size=16)
        origin = env.scene.env_origins[0].cpu().numpy()
        with torch.no_grad():
            for step in range(round(args.seconds * 50)):
                if step % 2 == 0:
                    actual.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                    actual.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                    qpos, qvel = cmd.library.ref_qpos_qvel(cmd.frame)
                    reference.qpos[:] = qpos[0].cpu().numpy()
                    reference.qpos[:3] += origin
                    reference.qvel[:] = qvel[0].cpu().numpy()
                    panels = []
                    for data in (actual, reference):
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = data.qpos[:3]
                        camera.lookat[2] = 0.8
                        renderer.update_scene(data, camera=camera)
                        panels.append(renderer.render().copy())
                    frame = Image.fromarray(np.concatenate(panels, axis=1))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1280, 32), fill=(15, 20, 28))
                    draw.text((12, 6), f'POLICY | {args.clip} | {step / 50:.2f} s', font=font, fill='white')
                    draw.text((652, 6), 'REFERENCE | anchor-observation reproduction', font=font, fill='white')
                    writer.append_data(np.asarray(frame))
                    if frames == 0:
                        frame.save(args.out.with_suffix('.png'))
                    frames += 1
                obs, _, term, tout, _ = env.step(policy(obs['policy'], obs['goal']))
                steps += 1
                if bool(term[0] | tout[0]):
                    failed = bool(term[0])
                    break
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        env.close()
    assert frames > 0
    result = dict(clip=args.clip,checkpoint=str(args.ckpt),video=str(args.out),frames=frames,fps=25,steps=steps,failed=failed)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
