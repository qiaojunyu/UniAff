import argparse
import pdb
import glob
import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import *


def get_part_mask_bbox(part_masks, view=True):
    handle_mask = (part_masks > 0).astype(np.uint8)
    contours, _ = cv2.findContours(handle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) > 4:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) > 4:
        while len(approx) > 4:
            distances = [np.linalg.norm(approx[i][0] - approx[(i + 1) % len(approx)][0]) for i in range(len(approx))]
            min_index = np.argmin(distances)
            approx = np.delete(approx, (min_index + 1) % len(approx), axis=0)
    if len(approx) == 4:
        sorted_points = approx.reshape(-1, 2)
    else:
        sorted_points = cv2.convexHull(approx).reshape(-1, 2)

    if sorted_points.shape[0] < 4:
        coords = np.column_stack(np.where(handle_mask > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        top_left = (x_min, y_min)
        top_right = (x_max, y_min)
        bottom_left = (x_min, y_max)
        bottom_right = (x_max, y_max)
        sorted_points = np.array([top_left, top_right, bottom_right, bottom_left])
    if view:
        plt.imshow(handle_mask, cmap='gray')
        plt.plot(np.append(sorted_points[:, 0], sorted_points[0, 0]),
                 np.append(sorted_points[:, 1], sorted_points[0, 1]), 'r-', lw=2)
        plt.scatter(sorted_points[:, 0], sorted_points[:, 1], c='blue', s=100)
        for i, point in enumerate(sorted_points):
            plt.text(point[0], point[1], f'({point[0]}, {point[1]})', color='yellow', fontsize=12,
                     ha='right' if i % 2 == 0 else 'left', va='top' if i < 2 else 'bottom')
        plt.title('Approximate Polygon with Four Corners')
        plt.show()
    print("sorted_points: ", sorted_points.shape)
    return sorted_points

def get_scaled_rotated_box(box, image_width=960, image_height=960, pad_x0=0, pad_y0=0, str_rep=True):
    # Unpack the original bounding box
    cx, cy, w, h, angle = box

    # Add the padding to the center coordinates
    cx_padded = cx + pad_x0
    cy_padded = cy + pad_y0

    # Scale the center coordinates
    scx = cx_padded / image_width
    scy = cy_padded / image_height

    # Scale the width and height
    sw = w / image_width
    sh = h / image_height

    # The angle remains the same since it's independent of the scale
    sangle = angle
    if str_rep and type(angle) is not str:
        sangle = "{:.2f}".format(angle)

    if str_rep:
        return "[{:.2f},{:.2f},{:.2f},{:.2f},{}]".format(scx, scy, sw, sh, sangle)

    return scx, scy, sw, sh, sangle


def normalize_and_round_angle(theta, granularity=5, range_start=0, range_end=360):
    # Normalize theta to be within [range_start, range_end)
    theta_normalized = (theta - range_start) % (range_end - range_start) + range_start

    # Round theta to the nearest granularity
    rounded_angle = round(theta_normalized / granularity) * granularity

    # Make sure the rounded angle is still within the range
    if rounded_angle == range_end:
        rounded_angle = range_start

    return rounded_angle / 360 * 3.1415
    
def find_minimum_rotated_bounding_box(part_masks):
    mask = (part_masks > 0).astype(np.uint8)
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rotated_rect = cv2.minAreaRect(largest_contour)

        center, size, angle = rotated_rect
        box_points = cv2.boxPoints(rotated_rect).astype(int)
        box_points = [(int(point[0]), int(point[1])) for point in box_points]

        x1 = float(box_points[0][0])
        y1 = float(box_points[0][1])
        x2 = float(box_points[1][0])
        y2 = float(box_points[1][1])
        x3 = float(box_points[2][0])
        y3 = float(box_points[2][1])
        x4 = float(box_points[3][0])
        y4 = float(box_points[3][1])

        if height > width:
            pad_x0 = int((height - width) / 2)
            pad_y0 = 0
            width = height
        else:
            pad_x0 = 0
            pad_y0 = int((width - height) / 2)
            height = width

        x_new = []
        for x in [x1, x2, x3, x4]:
            x = x + pad_x0
            x = x / width
            x_new.append(x)
        x1, x2, x3, x4 = x_new

        y_new = []
        for y in [y1, y2, y3, y4]:
            y = y + pad_y0
            y = y / height
            y_new.append(y)
        y1, y2, y3, y4 = y_new

        # cx, cy, w, h, angle
        bbox = (center[0], center[1], size[0], size[1], angle)
        scx, scy, sw, sh, sangle = get_scaled_rotated_box(bbox, image_width=width, image_height=height, pad_x0=pad_x0,
                                                          pad_y0=pad_y0, str_rep=False)
        sangle = normalize_and_round_angle(sangle, granularity=5, range_start=0, range_end=180)
        center = (scx, scy)
        size = (sw, sh)
        return x1, y1, x2, y2, x3, y3, x4, y4, center, size, sangle

def visualize_minimum_rotated_bounding_box(image, box_points_center_size_angle):
    # mask = (part_masks > 0).astype(np.uint8)
    x1, y1, x2, y2, x3, y3, x4, y4, center, size, angle = box_points_center_size_angle
    x1 = int(x1 * image.shape[1])
    y1 = int(y1 * image.shape[0])
    x2 = int(x2 * image.shape[1])
    y2 = int(y2 * image.shape[0])
    x3 = int(x3 * image.shape[1])
    y3 = int(y3 * image.shape[0])
    x4 = int(x4 * image.shape[1])
    y4 = int(y4 * image.shape[0])
    cv2.drawContours(image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image with Rotated Bounding Box")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='object 3d')
    parser.add_argument('--object_cat', type=str, default="power_drill")
    parser.add_argument('--down_sample_number', type=int, default=0)
    parser.add_argument('--FPS_sample_number', type=int, default=20000)
    parser.add_argument('--down_sample_save_path', type=str, default="./uniaff_tool_final/")
    parser.add_argument('--render_data_path', type=str, default="=./render_test/")
    parser.add_argument('--urdf_path_root', type=str, default="./uniaff/tool/test_intra/")
    parser.add_argument('--all', type=int, default=1)
    parser.add_argument('--view_data', type=int, default=0)
    args = parser.parse_args()
    if args.all:
        object_cats = os.listdir(args.render_data_path)
    else:
        object_cats = [args.object_cat]
    for object_cat in object_cats:
        render_datas = glob.glob(args.render_data_path + object_cat + "/*")
        for render_data in render_datas:
            view_files = glob.glob(render_data + "/*/*")
            for file_path in view_files:
                print("*"*100)
                with open("{}/pose_annotation.json".format(file_path), "r") as fp:
                    pose_annotation = json.load(fp)
                    print("process file: {}".format(file_path))
                files = np.load(file_path + "/data.npz", allow_pickle=True)
                object_mask_ids = [files["object_mask_id"]]
                object_scale = files["model_scale"]
                camera_point_cloud, per_point_rgb, per_point_idx, camera2world_matrix, mask = get_scene_point_cloud(files, view_pcd=False)
                object_cat = file_path.split("/")[-3]
                object_id = file_path.split("/")[-2]
                view_id = file_path.split("/")[-1]
                pose_matrix = change_pose_matrix(files["object_pose"], pose_annotation)
                object_size = pose_annotation["object_size"]
                object_camera_pcds = camera_point_cloud[mask == files["object_mask_id"]]
                if object_camera_pcds.shape[0] < 1000:
                    continue
                object_world_pcds = pc_camera_to_world(object_camera_pcds, camera2world_matrix)
                center, extent_length, rot_matrix = create_minimum_axis_aligned_bbox(object_world_pcds)
                pose_matrix[:3, 3] = center

                x_vec = rot_matrix[:, 0]
                y_vec = rot_matrix[:, 1]
                z_vec = rot_matrix[:, 2]
                anno_x_vec = pose_matrix[:3, 0]
                anno_y_vec = pose_matrix[:3, 1]
                anno_z_vec = pose_matrix[:3, 2]

                new_anno_x_vec, new_anno_y_vec, new_anno_z_vec, match_indices = tanslation_bbox_to_anno(x_vec, y_vec, z_vec, anno_x_vec, anno_y_vec, anno_z_vec)
                pose_matrix[:3, 0] = new_anno_x_vec
                pose_matrix[:3, 1] = new_anno_y_vec
                pose_matrix[:3, 2] = new_anno_z_vec

                pose_x_end = pose_matrix[:3, 0] + pose_matrix[:3, 3]
                pose_y_end = pose_matrix[:3, 1] + pose_matrix[:3, 3]
                pose_z_end = pose_matrix[:3, 2] + pose_matrix[:3, 3]
                pose_center = pose_matrix[:3, 3]



                camera_pose_x_end = translate_pc_world_to_camera(pose_x_end, camera2world_matrix)
                camera_pose_y_end = translate_pc_world_to_camera(pose_y_end, camera2world_matrix)
                camera_pose_z_end = translate_pc_world_to_camera(pose_z_end, camera2world_matrix)
                camera_pose_center = translate_pc_world_to_camera(pose_center, camera2world_matrix)

                camera_pose = np.eye(4)
                camera_pose[:3, 3] = camera_pose_center
                camera_pose[:3, 0] = (camera_pose_x_end - camera_pose_center) / np.linalg.norm(
                    camera_pose_x_end - camera_pose_center)
                camera_pose[:3, 1] = (camera_pose_y_end - camera_pose_center) / np.linalg.norm(
                    camera_pose_y_end - camera_pose_center)
                camera_pose[:3, 2] = (camera_pose_z_end - camera_pose_center) / np.linalg.norm(
                    camera_pose_z_end - camera_pose_center)

                object_center = camera_pose[:3, 3]
                object_x_size = extent_length[match_indices[0]]/2.0
                object_y_size = extent_length[match_indices[1]]/2.0
                object_z_size = extent_length[match_indices[2]]/2.0

                print("object size: ", object_size)
                line_x = camera_pose[:3, 0] * object_x_size
                line_y = camera_pose[:3, 1] * object_y_size
                line_z = camera_pose[:3, 2] * object_z_size
                camera_point_start = object_center - line_x - line_y - line_z
                camera_x_end = camera_point_start + line_x*2
                camera_y_end = camera_point_start + line_y*2
                camera_z_end = camera_point_start + line_z*2


                bound_max = camera_point_cloud.max(0)
                bound_min = camera_point_cloud.min(0)
                points_normalized = (camera_point_cloud - bound_min) / ((bound_max - bound_min))
                print("depth max: ", bound_min, " depth min: ", bound_max)

                camera_point_start = (camera_point_start - bound_min) / (bound_max - bound_min)
                camera_x_end = (camera_x_end - bound_min) / (bound_max - bound_min)
                camera_y_end = (camera_y_end - bound_min) / (bound_max - bound_min)
                camera_z_end = (camera_z_end - bound_min) / (bound_max - bound_min)

                pose_list = []
                pose_list.append([camera_point_start, camera_x_end])
                pose_list.append([camera_point_start, camera_y_end])
                pose_list.append([camera_point_start, camera_z_end])

                object_mask_id = files["object_mask_id"]
                object_masks = files["mask_map"]
                mesh_masks = files["mesh_mask_image"]

                image = files["rgb_image"]
                function_object_masks = (object_masks == object_mask_id).astype(np.uint8)
                if function_object_masks.sum() < 300:
                    print("no object")
                    continue
                coords = np.column_stack(np.where(function_object_masks > 0))
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                # object_mask_point = get_part_mask_bbox(function_object_masks)
                object_minimum_rotated_bounding_box = find_minimum_rotated_bounding_box(function_object_masks)
                # visualize_minimum_rotated_bounding_box(image, object_minimum_rotated_bounding_box)
                if camera_point_start.min() < 0 or camera_x_end.min() < 0 or camera_y_end.min() < 0 or camera_z_end.min() < 0:
                    print("camera_point_start:{}".format(camera_point_start))
                    print("camera_x_end:{}".format(camera_x_end))
                    print("camera_y_end:{}".format(camera_y_end))
                    print("camera_z_end:{}".format(camera_z_end))
                    camera_point_start = np.clip(camera_point_start, 0, 1)
                    camera_x_end = np.clip(camera_x_end, 0, 1)
                    camera_y_end = np.clip(camera_y_end, 0, 1)
                    camera_z_end = np.clip(camera_z_end, 0, 1)
                    # pdb.set_trace()
                # anno
                anno_files = {"camera_point_start": camera_point_start.tolist(),
                              "camera_x_end": camera_x_end.tolist(),
                              "camera_y_end": camera_y_end.tolist(),
                              "camera_z_end": camera_z_end.tolist(),
                              "object_four_point": list(object_minimum_rotated_bounding_box),
                              "object_2d_bbox": [float(x_min), float(x_max), float(y_min), float(y_max)],
                              }
                object_part_mask = np.zeros_like(object_masks)
                object_part_mask[object_masks == object_mask_id] = mesh_masks[object_masks == object_mask_id]
                part_ids = np.unique(mesh_masks[object_masks == object_mask_id])
                object_parts = files["object_part"]
                if object_parts != part_ids.shape[0]:
                    print("part num not match: ", object_parts, part_ids.shape[0])
                    continue
                for id, part_id in enumerate(part_ids):
                    coords = np.column_stack(np.where(object_part_mask == part_id))
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    # part_mask_point = get_part_mask_bbox(object_part_mask == part_id)
                    part_minimum_rotated_bounding_box = find_minimum_rotated_bounding_box(object_part_mask == part_id)
                    # visualize_minimum_rotated_bounding_box(image, part_minimum_rotated_bounding_box)
                    if id == 0:
                        anno_files.update({"part_handle_2d_bbox": [float(x_min), float(x_max), float(y_min), float(y_max)]})
                        anno_files.update({"part_handle_four_point": list(part_minimum_rotated_bounding_box)})
                    if id == 1:
                        anno_files.update({"part_function_2d_bbox": [float(x_min), float(x_max), float(y_min), float(y_max)]})
                        anno_files.update({"part_function_four_point": list(part_minimum_rotated_bounding_box)})
                

                # print("*"*100)
                print("file path: {}".format(file_path))
                save_path = "{}/{}/{}".format(args.down_sample_save_path, object_cat, "image")
                print("save file: {}".format(save_path))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                rgb_image = Image.open(file_path + "/image.png")
                rgb_image.save(save_path+"/{}_{}.png".format(object_id, view_id))

                save_path = "{}/{}/{}".format(args.down_sample_save_path, object_cat, "mask")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                print("mask ids: ", np.unique(object_part_mask))
                image = Image.fromarray(object_part_mask)
                image.save(save_path + "/{}_{}.png".format(object_id, view_id))


                # save_path = "{}/{}/{}".format(args.down_sample_save_path, object_cat, "pcd")
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                #
                # np.save(save_path + "/{}_{}.npy".format(object_id, view_id), points_normalized)

                save_path = "{}/{}/{}".format(args.down_sample_save_path, object_cat, "pose_anno")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(save_path + "/{}_{}.json".format(object_id, view_id), "w") as fp:
                    json.dump(anno_files, fp)


                save_path = os.path.join(args.down_sample_save_path, object_cat, "view_parameter")
                os.makedirs(save_path, exist_ok=True)
                data = {"bound_max": bound_max.tolist(),
                        "bound_min": bound_min.tolist(),
                        "camera_intrinsic": files['camera_intrinsic'].tolist()}
                with open(save_path + "/{}_{}.json".format(object_id, view_id), "w") as fp:
                    json.dump(data, fp)

                if args.view_data:
                    view_rigid_object_pose(points_normalized, per_point_rgb, pose_list)
                    view_affordance_image_4pt(rgb_image, part_minimum_rotated_bounding_box)













