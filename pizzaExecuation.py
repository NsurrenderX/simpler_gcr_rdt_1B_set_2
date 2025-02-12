from frankaRe3Curobo import frankaRe3

import requests
import os
import yaml
import base64
import numpy as np
import json
import cv2
import imageio

from PIL import Image
from scipy.spatial.transform import Rotation as R

def convert_euler_to_rotation_matrix(euler):
    """
    Convert Euler angles (rpy) to rotation matrix (3x3).
    """
    quat = R.from_euler('xyz', euler).as_matrix()
    
    return quat

def compute_ortho6d_from_rotation_matrix(matrix):
    # The ortho6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    
    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix

def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler('xyz', degrees=False)
    
    return euler

def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.stack((i, j, k), axis=1)
    return out

def convert_to_10D_action(action):
    """
    Convert 7D action to 10D action.
    """
    trans = action[0]
    rot = action[1]
    gripper = action[2]
    
    format_action = np.zeros(10).astype(np.float32)
    rot_mat = R.from_quat(rot).as_matrix()
    
    for i in range(3):
        format_action[i] = trans[i]
    ortho6d = compute_ortho6d_from_rotation_matrix(np.expand_dims(rot_mat, axis=0))[0]
    
    for i in range(6):
        format_action[i+3] = ortho6d[i]
        
    format_action[9] = gripper
    
    if gripper <= 0.067:
        format_action[9] = 0.0
    else:
        format_action[9] = 1
    return format_action

def convert_to_7D_action(action):
    """
    Convert 10D action to 7D action.
    """
    trans = action[0:3]
    rot = action[3:9]
    gripper = action[9]

    rot_mat = compute_rotation_matrix_from_ortho6d(np.expand_dims(rot, axis=0))[0]
    euler = convert_rotation_matrix_to_euler(rot_mat)

    format_action = np.zeros(7).astype(np.float32)
    for i in range(3):
        format_action[i] = trans[i]
    for i in range(3):
        format_action[i+3] = euler[i]
    format_action[6] = gripper

    return format_action

def decode_b64_image(b64image):
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    image_cv2 = image_cv2[40:720,200:880,:]
    return image_cv2

def get_image(camera_url = "http://127.0.0.1:5000/get_full"):
    resp = requests.get(url=camera_url)
    images = resp.json()
    images = images
    print("Image received")
    return images

def exec_robot(fr3, idx, action):
    print("=======================================")
    print("Executing action: ")
    # print("Current Step: ", idx)
    action_queue.append(action)
    action_to_exec = convert_to_7D_action(action)
    print("Detailed action: ", action_to_exec)
    action = [action_to_exec[:3], action_to_exec[3:6], action_to_exec[6]]
    success = 1
    if config['padding'] - action_to_exec[0] <= 1e-5 or action_to_exec[0] > 1.5*np.pi:
        print("Padding Value detected, aboritng this execution")
        success = 0
        return success
    fr3.exec_action(action, duration=1.5, buffer_time=0.1, ignore_virtual_walls = True)
    current_width = fr3.arm.get_gripper_width()
    tgt_gripper = action[2]
    # if current_width >= 0.015:
    #     current_gripper = 1
    # else:
    #     current_gripper = -1
    # if tgt_gripper > 0.001:
    #     gripper = -1
    # else:
    #     gripper = 1
    # if current_gripper * gripper == -1:
    #     if current_gripper == 1:
    #         print("Closing Gripper")
    #         fr3.arm.goto_gripper(0.0, speed=1.5)
    #     else:
    #         print("Opening Gripper")
    #         fr3.arm.goto_gripper(0.078, speed=1.5)
    
    trans, rot = fr3.get_current_pose()
    gripper = fr3.arm.get_gripper_width()
    if gripper <= 0.067:
        gripper = 0.0
    else:
        gripper = 1
    
    # rot_euler = R.from_quat(rot).as_euler('xyz', degrees=False)
    # ortho6d = compute_ortho6d_from_rotation_matrix(np.expand_dims(rot_mat, axis=0))[0]
    
    state_list = [trans, rot, gripper]
    state = convert_to_10D_action(state_list)
    print("Current State: ", state)

    print("=======================================\n")

    state_queue.append(state.tolist())
    
    image_queue.append(get_image()['k4a_0'])

    return success

def control_loop(fr3, task_id, exec_per_step, max_step):
    global state_queue, action_queue, image_queue, config
    step = 0
    while step < max_step:
        # images = get_image()
        # image_queue.append(images['k4a_0'])
        # joint = fr3.arm.get_joints()
        # gripper = fr3.arm.get_gripper_width()
        # joint_queue.append(joint.tolist())
        # gripper_queue.append(gripper)
        instance_data = {
            'action': action_queue,
            'state': state_queue,
            'image': [image_queue[-2], image_queue[-1]] if len(image_queue) > 1 else [image_queue[-1]],
            'task_id': task_id
        }
        action_json = requests.post(url = config['url'], json=instance_data)
        actions = action_json.json()['actions']

        for idx in range(exec_per_step):
            print("Current Step: ", step)
            action = actions[idx]
            success = exec_robot(fr3, idx, action)
            step += success



if __name__ == "__main__":
    # Load config
    yaml_path = "configs/pizza.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Robot Init
    fr3 = frankaRe3(config['franka']['config'])

    # Init State Data
    state_queue = []
    action_queue = []
    first_trans, first_rotate = fr3.get_current_pose()
    first_gripper = fr3.arm.get_gripper_width()
    
    first_state_list = [first_trans, first_rotate, first_gripper]
    first_state = convert_to_10D_action(first_state_list)
    
    zero_action = np.zeros(10).astype(np.float32)

    state_queue.append(first_state.tolist())
    action_queue.append(zero_action.tolist())
    # action_queue.append(first_gripper)

    # Init Image Data
    images = get_image()
    image_queue = []
    image_queue.append(images['k4a_0'])

    control_loop(fr3, config['task_id'], config['exec']['inference_per_step'], config['exec']['max_step'])

    img_IMAGE_queue = []
    for img in image_queue:
        img_IMAGE_queue.append(Image.fromarray(decode_b64_image(img)))

    # Save Data
    imageio.mimsave('/datahdd_8T/vla_pizza/ours/gifs/latest.gif', img_IMAGE_queue, "GIF", fps=10)



