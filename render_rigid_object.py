import pdb
import sapien.core as sapien
import numpy as np
from utils import *
import os
import argparse
import glob
from transforms3d.euler import mat2euler
import matplotlib.pyplot as plt
from sapien.utils import Viewer
from scipy.spatial.transform import Rotation

def set_scene(cam_pos, width=800, height=800, obj_file_path=None, scale=0.001, ray_tracing=False, random_color=True, debug=False):
    assert os.path.exists(obj_file_path)
    sim = sapien.Engine()
    renderer = sapien.SapienRenderer()
    sim.set_renderer(renderer)
    if ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 64
        sapien.render_config.rt_use_denoiser = True
    renderer = sapien.SapienRenderer()
    sim.set_renderer(renderer)
    scene_config = sapien.SceneConfig()
    scene = sim.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    color_point = 1.0
    if random_color:
        color = 0.3 * np.random.rand() + 0.5
        color_point = 0.55 * np.random.rand() + 0.45
        scene.set_ambient_light([color, color, color])
        scene.add_directional_light([0, 0.5, -1], color=[3.0, 3.0, 3.0],
                                    shadow=True, scale=2.0, shadow_map_size=4096
                                    )
    else:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 0.5, -1], color=[3.0, 3.0, 3.0],
                                    shadow=True, scale=2.0, shadow_map_size=4096
                                    )
    if debug:
        viewer = Viewer(renderer)  # Create a viewer (window)
        viewer.set_scene(scene)  # Bind the viewer and the scene
        # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # The principle axis of the camera is the x-axis
        viewer.set_camera_xyz(x=-1.5, y=0, z=0.8)
        # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # The camera now looks at the origin
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    scene.add_point_light([1, 2, 2], [color_point, color_point, color_point], shadow=True)
    scene.add_point_light([1, -2, 2], [color_point, color_point, color_point], shadow=True)
    scene.add_point_light([-1, 0, 2], [color_point, color_point, color_point], shadow=True)
    path = "./background/glb/kitchen_small.glb"
    quat = Rotation.from_rotvec([1.57, 0.0, 0]).as_quat()[[3, 0, 1, 2]]
    pose = sapien.Pose([0, -0.0, 0.0], quat)
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(path))
    builder.add_collision_from_file("./background/glb/5.obj")
    visual_bg = builder.build_kinematic()
    visual_bg.set_pose(pose)
    for _ in range(100):
        scene.step()
    model_scale = float(scale)
    back_objects = os.listdir("./background/object/")
    selected_objects = np.random.choice(back_objects, np.random.randint(3, 9), replace=False)
    for obj in selected_objects:
        visual_path = "./background/object/" + obj + "/textured.obj"
        collision_path = "./background/object/" + obj + "/coacd_collision.obj"
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(visual_path, scale=[0.001]*3)
        builder.add_collision_from_file(
                    filename=collision_path,
                    scale=[0.001]*3,
                    density=100,
                )
        back_actor = builder.build()
        quat = Rotation.from_rotvec([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]).as_quat()[[3, 0, 1, 2]]
        back_actor.set_pose(sapien.Pose([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0.10, 0.25)], quat))
        for _ in range(10):
            scene.step()
            if debug:
                scene.update_render()  # Update the world to the renderer
                viewer.render()
    for _ in range(100):
        scene.step()
        if debug:
            scene.update_render()  # Update the world to the renderer
            viewer.render()
    object_part = 2
    builder = scene.create_actor_builder()
    builder.add_collision_from_file(
        filename=obj_file_path + "/textured_handle.obj",
        scale=[model_scale]*3,
        density=100,
    )
    builder.add_visual_from_file(filename=obj_file_path + "/textured_handle.obj", scale=[model_scale] * 3)
    builder.add_collision_from_file(
        filename=obj_file_path + "/textured_function.obj",
        scale=[model_scale] * 3,
        density=100,
    )
    builder.add_visual_from_file(filename=obj_file_path + "/textured_function.obj", scale=[model_scale] * 3)
    if os.path.exists(obj_file_path + "/textured_other.obj"):
        builder.add_collision_from_file(
            filename=obj_file_path + "/textured_other.obj",
            scale=[model_scale] * 3,
            density=100,
        )
        builder.add_visual_from_file(filename=obj_file_path + "/textured_other.obj", scale=[model_scale] * 3)
        object_part = 3
    actor = builder.build()
    quat = Rotation.from_rotvec([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]).as_quat()[[3, 0, 1, 2]]
    actor.set_pose(sapien.Pose([0.0, 0.0, np.random.uniform(0.1, 0.2)], quat))
    assert actor, "mesh not loaded."
    for _ in range(100):
        scene.step()
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera_1",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=width,
        height=height,
        fovx=np.deg2rad(35.0),
        fovy=np.deg2rad(35.0),
        near=0.1,
        far=100.0,
    )
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose(mat44))
    actor_pose = actor.get_pose()
    position = actor_pose.p
    orientation = actor_pose.q
    if abs(position[1]) - 0.15 > 0 or abs(orientation[0]) - 0.15 > 0:
        print("actor pose: {}".format(actor.get_pose()))
        actor.set_pose(sapien.Pose([0.0, 0.0, 0.10], q=orientation))
        for _ in range(100):
            scene.step()
    scene.update_render()
    camera.take_picture()
    print("actor pose: {}".format(actor.get_pose()))
    if debug:
        while not viewer.closed:  # Press key q to quit
            scene.step()  # Simulate the world
            scene.update_render()  # Update the world to the renderer
            viewer.render()
    return scene, camera, actor, model_scale, object_part

def render_an_image(cam_pos,  width=800, height=800, obj_file_path=None,  ray_tracing=False, render_image=True, render_depth=True, debug=False):
    pose_anno = load_json(obj_file_path + "/model_annnotation.json")
    scale = pose_anno["scale_size"]
    scene, camera, robot, model_scale, object_part = set_scene(cam_pos, width, height, obj_file_path, scale, ray_tracing=ray_tracing, debug=debug)
    intrinsic_matrix = camera.get_intrinsic_matrix()
    result = {"camera_intrinsic": intrinsic_matrix, "model_scale": model_scale, "object_part": object_part}
    pose_anno = load_json(obj_file_path + "/model_annnotation.json")
    result.update({"pose_annotation": pose_anno})
    if render_image:
        rgba = camera.get_float_texture('Color')
        rgb = rgba[:, :, :3]
        rgb_img = (rgb * 255).clip(0, 255).astype("uint8")
        result["rgb_image"] = rgb_img
        if debug:
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.show()
    if render_depth:
        position = camera.get_float_texture('Position') # [H, W, 4]
        depth_map = -position[..., 2]
        camera2world_matrix = get_camera_pos_mat(camera)
        valid_depth_mask = position[..., 3] < 1
        result["valid_depth_mask"] = valid_depth_mask
        result["depth_map"] = depth_map
        result["camera2world_matrix"] = camera2world_matrix
        print("camera2world_matrix: {}".format(camera2world_matrix[:3, 3]))
    # pose_annotation = load_json(obj_file_path + "/model_annnotation.json")
    object_pose = get_object_pose(robot)
    mesh_mask_image, part_mask_image = get_seg_mask(camera)
    result["mesh_mask_image"] = mesh_mask_image
    if debug:
        plt.imshow(mesh_mask_image)
        plt.axis('off')
        plt.show()
    result["part_mask_image"] = part_mask_image
    result["object_mask_id"] = robot.get_id()
    if debug:
        plt.imshow(part_mask_image)
        plt.axis('off')
        plt.show()
    result["object_pose"] = object_pose
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='object 3d')
    parser.add_argument('--view_num', type=int, default=15)
    parser.add_argument('--ray_tracing', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--object_type', type=str, default="test_intra")
    parser.add_argument('--file_path', type=str, default="./uniaff/tool/")
    parser.add_argument('--save_path', type=str, default="./render_test/")
    args = parser.parse_args()
    save_path = args.save_path + args.object_type + "/"
    object_cats = os.listdir("{}/{}".format(args.file_path, args.object_type))
    for object_cat in object_cats:
        files = glob.glob("{}/{}/{}/*".format(args.file_path, args.object_type, object_cat))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        for file in files:
            for view in range(args.view_num):
                shape_id = file.split("/")[-1]
                theta = np.random.uniform(0, 50)
                phi = np.random.uniform(0, 360)
                distance = np.random.uniform(0.80, 1.35)
                camera_pos = get_around_cam_pos(theta=-30, phi=0, distance=distance)
                render_result = render_an_image(cam_pos=camera_pos, obj_file_path=file, ray_tracing=args.ray_tracing, debug=args.debug)
                save_rigid_render_data(save_path + object_cat, view, shape_id, render_result, npz_file=True, image_save=True, mask_save=True)





