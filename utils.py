import math
import numpy as np
import gzip
import json
import os
from PIL import Image, ImageColor
import copy
import open3d as o3d
import json
import pdb
import ast
import cv2
import matplotlib.pyplot as plt

def get_cam_pos(theta, phi, distance):
    x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
    y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
    z = math.cos(math.pi / 180 * theta) * distance
    return np.array([x, y, z])

def get_around_cam_pos(theta, phi, distance):
    x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
    y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
    z = math.cos(math.pi / 180 * theta) * distance
    return np.array([x, y, z])

def load_json(filename):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def view_rigid_object_pose(pcds, rgbs=None, pose_lists=None):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcds)
    if rgbs is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgbs)
    lines_pcds = []
    if pose_lists is not None:
        polygon_points = np.array([pose_lists[0][0].tolist(), pose_lists[0][1].tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([np.array([1, 0, 0])])
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)

        polygon_points = np.array([pose_lists[1][0].tolist(), pose_lists[1][1].tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([np.array([0, 1, 0])])
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)

        polygon_points = np.array([pose_lists[2][0].tolist(), pose_lists[2][1].tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([np.array([0, 0, 1])])
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
    o3d.visualization.draw_geometries([cloud, *lines_pcds, axis_pcd])


def get_camera_pos_mat(camera):
    Rtilt = camera.get_model_matrix()
    Rtilt_rot = Rtilt[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = Rtilt[:3, 3]
    cam2_wolrd = np.eye(4)
    cam2_wolrd[:3, :3] = Rtilt_rot
    cam2_wolrd[:3, 3] = Rtilt_trl
    return cam2_wolrd

def get_object_pose(actor):
    object_pose = actor.pose.to_transformation_matrix()
    return object_pose

def translate_pc_world_to_camera(point_cloud, extrinsic):
    extr_inv = np.linalg.inv(extrinsic)
    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ point_cloud.T).T + T
    return pc

def get_seg_mask(camera):
    seg_labels = camera.get_uint32_texture("Segmentation")  # [H, W, 4]
    mesh_mask_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    part_mask_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    return mesh_mask_image, part_mask_image

def pc_camera_to_world(pc, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    pc = (R @ pc.T).T + T
    return pc

def create_minimum_axis_aligned_bbox(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_minimal_oriented_bounding_box()
    center = np.asarray(obb.center)
    extent = np.asarray(obb.extent)
    rot_matrix = np.array(obb.R)
    return center, extent, rot_matrix

def change_pose_matrix(pose_matrix, object_annotation):
    new_pose_mat = copy.deepcopy(pose_matrix)
    direction_axis = new_pose_mat[:3, object_annotation["direction_axis"]["axis"]] * object_annotation["direction_axis"]["axis_sign"]
    approach_axis = new_pose_mat[:3, object_annotation["approach_axis"]["axis"]] * object_annotation["approach_axis"]["axis_sign"]
    z_axis = np.cross(direction_axis, approach_axis)
    R_Mat = np.zeros((3, 3))
    R_Mat[:, 0] = direction_axis
    R_Mat[:, 1] = approach_axis
    R_Mat[:, 2] = z_axis
    new_pose_mat[:3, :3] = R_Mat
    return new_pose_mat

def save_rigid_render_data(save_path, view, save_id, files, npz_file=True, image_save=False, mask_save=False):
    save_path = save_path + "/" + save_id + "/"
    os.makedirs(save_path, exist_ok=True)
    file_path = save_path + str(view)
    if npz_file:
        os.makedirs(file_path, exist_ok=True)
        np.savez(file_path + "/data.npz",
                 rgb_image=files["rgb_image"],
                 depth_map=files["depth_map"],
                 mask_map=files["part_mask_image"],
                 mesh_mask_image=files["mesh_mask_image"],
                 camera_intrinsic=files["camera_intrinsic"],
                 camera2world_matrix=files["camera2world_matrix"],
                 object_pose=files["object_pose"],
                 object_mask_id=files["object_mask_id"],
                 model_scale=files["model_scale"],
                 object_part=files["object_part"]
                 )
        print("mask ids: ", np.unique(files["mesh_mask_image"]))
        os.makedirs(file_path, exist_ok=True)

    with open(file_path + "/pose_annotation.json", "w") as f:
        json.dump(files["pose_annotation"], f)

    if image_save:
        os.makedirs(file_path, exist_ok=True)
        new_image = Image.fromarray(files["rgb_image"])
        new_image.save(file_path + "/image.png")

    if mask_save:
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array(
            [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
        )
        os.makedirs(file_path, exist_ok=True)
        new_image = Image.fromarray(color_palette[files["mesh_mask_image"]])
        new_image.save(file_path + "/mesh_mask_image.png")
        new_image = Image.fromarray(color_palette[files["part_mask_image"]])
        new_image.save(file_path + "/part_mask_image.png")

def view_point_clouds(point_cloud, rgbs=None,  camera_pose=None):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud)
    if rgbs is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgbs)
    if camera_pose is not None:
        lines_pcds = []
        polygon_points = np.array([camera_pose[:3, 3].tolist(), (camera_pose[:3, 3] + camera_pose[:3, 0]*0.1).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([np.array([1, 0, 0])])
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
        polygon_points = np.array([camera_pose[:3, 3].tolist(), (camera_pose[:3, 3] + camera_pose[:3, 1]*0.1).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([np.array([0, 1, 0])])
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
        polygon_points = np.array([camera_pose[:3, 3].tolist(), (camera_pose[:3, 3] + camera_pose[:3, 2]*0.1).tolist()])
        lines = [[0, 1]]
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector(lines)
        lines_pcd.colors = o3d.utility.Vector3dVector([np.array([0, 0, 1])])
        lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
        lines_pcds.append(lines_pcd)
        o3d.visualization.draw_geometries([cloud, *lines_pcds,axis_pcd])
    else:
        o3d.visualization.draw_geometries([cloud, axis_pcd])

def tanslation_bbox_to_anno(x_vec, y_vec, z_vec, anno_x_vec, anno_y_vec, anno_z_vec):

    cosine_x_similarity = np.dot(anno_x_vec, x_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(anno_x_vec))
    cosine_y_similarity = np.dot(anno_x_vec, y_vec) / (np.linalg.norm(y_vec) * np.linalg.norm(anno_x_vec))
    cosine_z_similarity = np.dot(anno_x_vec, z_vec) / (np.linalg.norm(z_vec) * np.linalg.norm(anno_x_vec))

    cosine_similarity = np.array([cosine_x_similarity, cosine_y_similarity, cosine_z_similarity])
    indice = np.argmax(np.abs(cosine_similarity))
    pcd_bbox = [x_vec, y_vec, z_vec]
    match_indices = [indice]
    if cosine_similarity[indice] > 0:
        new_anno_x_vec = pcd_bbox[indice]
    else:
        new_anno_x_vec = -pcd_bbox[indice]

    cosine_x_similarity = np.dot(anno_y_vec, x_vec) / (np.linalg.norm(anno_y_vec) * np.linalg.norm(x_vec))
    cosine_y_similarity = np.dot(anno_y_vec, y_vec) / (np.linalg.norm(anno_y_vec) * np.linalg.norm(y_vec))
    cosine_z_similarity = np.dot(anno_y_vec, z_vec) / (np.linalg.norm(anno_y_vec) * np.linalg.norm(z_vec))
    cosine_similarity = np.array([cosine_x_similarity, cosine_y_similarity, cosine_z_similarity])
    indice = np.argmax(np.abs(cosine_similarity))
    match_indices.append(indice)
    if cosine_similarity[indice] > 0:
        new_anno_y_vec = pcd_bbox[indice]
    else:
        new_anno_y_vec = -pcd_bbox[indice]

    cosine_x_similarity = np.dot(anno_z_vec, x_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(anno_z_vec))
    cosine_y_similarity = np.dot(anno_z_vec, y_vec) / (np.linalg.norm(y_vec) * np.linalg.norm(anno_z_vec))
    cosine_z_similarity = np.dot(anno_z_vec, z_vec) / (np.linalg.norm(z_vec) * np.linalg.norm(anno_z_vec))
    cosine_similarity = np.array([cosine_x_similarity, cosine_y_similarity, cosine_z_similarity])
    indice = np.argmax(np.abs(cosine_similarity))
    match_indices.append(indice)
    if cosine_similarity[indice] > 0:
        new_anno_z_vec = pcd_bbox[indice]
    else:
        new_anno_z_vec = -pcd_bbox[indice]
    z_direction = np.cross(new_anno_x_vec, new_anno_y_vec)
    if np.dot(z_direction, new_anno_z_vec) < 0:
        new_anno_z_vec = -new_anno_z_vec
        print("change z direction now")
    return new_anno_x_vec, new_anno_y_vec, new_anno_z_vec, match_indices

def get_scene_point_cloud(files, world=False, view_pcd=False, eps=0.006):
    rgb_image = files['rgb_image']
    depth_map = files['depth_map']
    mask_map = files["mask_map"]
    camera2world_matrix = files['camera2world_matrix']
    width, height = rgb_image.shape[0], rgb_image.shape[1]
    K = np.array(files['camera_intrinsic']).reshape(3, 3)
    y_coords, x_coords = np.indices((height, width))
    z_new = depth_map.astype(float)
    valid_mask = abs(depth_map) > eps
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    z_new = z_new[valid_mask]
    x_new = (x_coords - K[0, 2]) * z_new / K[0, 0]
    y_new = (y_coords - K[1, 2]) * z_new / K[1, 1]
    point_cloud = np.stack((x_new, y_new, z_new), axis=-1)
    per_point_rgb = rgb_image[y_coords, x_coords] / 255.0
    per_point_idx = np.stack((y_coords, x_coords), axis=-1)
    camera_point_cloud = np.array(point_cloud)
    per_point_rgb = np.array(per_point_rgb)
    per_point_idx = np.array(per_point_idx)
    if world:
        world_point_cloud = pc_camera_to_world(camera_point_cloud, camera2world_matrix)
        if view_pcd:
            view_point_clouds(world_point_cloud, per_point_rgb, camera_pose=camera2world_matrix)
        return world_point_cloud, per_point_rgb, per_point_idx
    if view_pcd:
        view_point_clouds(camera_point_cloud, per_point_rgb)
    return camera_point_cloud, per_point_rgb, per_point_idx, camera2world_matrix, mask_map[valid_mask]



def view_bbox_3Dt2D(view_parameters, image, bbox_3d_points_cams=None, joints=None, save_path=None):
    intrinsics = np.array(view_parameters["camera_intrinsic"]).reshape(3, 3)
    bound_max = np.array(view_parameters["bound_max"])
    bound_min = np.array(view_parameters["bound_min"])
    image = np.array(image)
    vis_image_3d = image.copy()

    if joints is not None:
        joints = np.array(joints)
        joints = joints*(bound_max - bound_min) + bound_min
        points_2d = []
        for point in joints:
            point = [point[0] / point[2], point[1] / point[2]]
            pixel_x = int(point[0] * intrinsics[0, 0] + intrinsics[0, 2])
            pixel_y = int(point[1] * intrinsics[1, 1] + intrinsics[1, 2])
            points_2d.append([pixel_x, pixel_y])
        points_2d = np.array(points_2d, dtype=np.int32)
        vis_image_3d = cv2.arrowedLine(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[1]), (0, 200, 200), 5)
    if bbox_3d_points_cams is not None:
        points_2d = []
        bbox_3d_points_cam = np.array(bbox_3d_points_cams)
        bbox_3d_points_cam = bbox_3d_points_cam * (bound_max - bound_min) + bound_min
        for point in bbox_3d_points_cam:
            point = [point[0] / point[2], point[1] / point[2]]
            pixel_x = int(point[0] * intrinsics[0, 0] + intrinsics[0, 2])
            pixel_y = int(point[1] * intrinsics[1, 1] + intrinsics[1, 2])
            points_2d.append([pixel_x, pixel_y])
        points_2d = np.array(points_2d, dtype=np.int32)
        # Draw 3D bounding box: 8 points
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[1]), (0, 0, 255), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[3]), (0, 255, 0), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[4]), (255, 0, 0), 2)

        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[1]), tuple(points_2d[2]), (200, 200, 200), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[1]), tuple(points_2d[5]), (200, 200, 200), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[3]), tuple(points_2d[2]), (200, 200, 200), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[3]), tuple(points_2d[7]), (200, 200, 200), 2)

        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[4]), tuple(points_2d[5]), (200, 200, 200), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[4]), tuple(points_2d[7]), (200, 200, 200), 2)

        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[6]), tuple(points_2d[2]), (200, 200, 200), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[6]), tuple(points_2d[5]), (200, 200, 200), 2)
        vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[6]), tuple(points_2d[7]), (200, 200, 200), 2)

    if save_path is not None:
        cv2.imwrite(save_path, vis_image_3d)
        print("finished save: ", save_path)
    else:
        cv2.imshow("test", vis_image_3d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def view_affordance_image(image, bbox, save_path=None):
    plt.imshow(image)
    x0, x1, y0, y1 = bbox
    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
    if save_path is not None:
        plt.savefig(save_path + "vis_affordance.png")
        plt.close()
    else:
        plt.show()


def view_affordance_image_4pt(image, bbox, save_path=None):
    plt.imshow(image)
    width, height = image.size
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    plt.plot([x1 * width, x2 * width, x3 * width, x4 * width, x1 * width],
             [y1 * height, y2 * height, y3 * height, y4 * height, y1 * height])
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path + "vis_affordance.png")
        plt.close()
    else:
        plt.show()


    