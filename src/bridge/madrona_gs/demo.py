import torch

import genesis as gs


def main():
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=(8.5, 0.0, 4.5),
            camera_lookat=(3.0, 0.0, 0.5),
            camera_fov=50,
        ),
        rigid_options=gs.options.RigidOptions(enable_collision=False, gravity=(0, 0, 0)),
        renderer = gs.options.renderers.BatchRenderer(
            use_rasterizer=True,
            batch_render_res=(1920, 1080),
        ),
    )

    ########################## materials ##########################

    ########################## entities ##########################
    # floor
    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -0.5),
        ),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )

    # user specified external color texture
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -3, 0.0),
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    # user specified color (using color shortcut)
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -1.8, 0.0),
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # smooth shortcut
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -0.6, 0.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(0.6, 0.8, 1.0),
        ),
    )
    # Iron
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 0.6, 0.0),
        ),
        surface=gs.surfaces.Iron(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Gold
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 1.8, 0.0),
        ),
        surface=gs.surfaces.Gold(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Glass
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 3.0, 0.0),
        ),
        surface=gs.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Opacity
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(2.0, -3, 0.0),
        ),
        surface=gs.surfaces.Smooth(color=(1.0, 1.0, 1.0, 0.5)),
    )
    # asset's own attributes
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -2.3, 0.0),
        ),
    )
    # override asset's attributes
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -1.0, 0.0),
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            )
        ),
    )
    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        pos=(8.5, 0.0, 4.5),
        lookat=(3.0, 0.0, 0.5),
        fov=50,
        GUI=True,
        spp=512,
    )
    scene.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[1.0, 1.0, -2.0],
        directional=1,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5
    )
    scene.build()

    ########################## forward + backward twice ##########################
    scene.reset()
    horizon = 10

    for i in range(horizon):
        scene.step()
        rgb, depth, _, _ = scene.batch_render()
        output_rgb_and_depth('img_output/test', rgb, depth, i)


import os
import cv2
import numpy as np

# TODO: Dump image faster, e.g., asynchronously or generate a video instead of saving images.
def output_rgb(output_dir, rgb, i_env, i_cam, i_step):
    rgb = rgb.cpu().numpy()[i_env, i_cam]
    cv2.imwrite(f'{output_dir}/rgb_env{i_env}_cam{i_cam}_{i_step:03d}.png', rgb)

def output_depth(output_dir, depth, i_env, i_cam, i_step):
    depth = depth.cpu().numpy()[i_env, i_cam]
    depth = np.asarray(depth)
    depth = np.clip(depth, 0, 100)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/depth_env{i_env}_cam{i_cam}_{i_step:03d}.png', depth_uint8)

def output_rgb_and_depth(output_dir, rgb, depth, i_step):
    bgr = rgb[..., [2, 1, 0]]
    # loop over the first and second dimension of rgb and depth
    for i_env in range(bgr.shape[0]):
        for i_cam in range(bgr.shape[1]):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_rgb(output_dir, bgr, i_env, i_cam, i_step)
            output_depth(output_dir, depth, i_env, i_cam, i_step)

def output_rgb_single_cam(output_dir, rgb, i_env, i_step, cam_idx):
    rgb = rgb.cpu().numpy()[i_env]
    cv2.imwrite(f'{output_dir}/rgb_env{i_env}_cam{cam_idx}_{i_step:03d}.png', rgb)

def output_depth_single_cam(output_dir, depth, i_env, i_step, cam_idx):
    depth = depth.cpu().numpy()[i_env]
    depth = np.asarray(depth)
    depth = np.clip(depth, 0, 100)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/depth_env{i_env}_cam{cam_idx}_{i_step:03d}.png', depth_uint8)

def output_rgb_and_depth_single_cam(output_dir, rgb, depth, i_step, cam_idx):
    bgr = rgb[..., [2, 1, 0]]
    # loop over the first and second dimension of rgb and depth
    for i_env in range(bgr.shape[0]):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_rgb_single_cam(output_dir, bgr, i_env, i_step, cam_idx)
        output_depth_single_cam(output_dir, depth, i_env, i_step, cam_idx)


if __name__ == "__main__":
    main()
