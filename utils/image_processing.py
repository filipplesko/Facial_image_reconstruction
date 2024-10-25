import os
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw
import mediapipe as mp
import random
import sys

# Initialize MediaPipe FaceMesh as a global object
mp_face_mesh = mp.solutions.face_mesh
face_mesh = None  # Will be initialized in a function later

# Function to initialize the FaceMesh model
def initialize_face_mesh():
    global face_mesh
    if face_mesh is None:
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Function to calculate face yaw (left/right rotation)
def calculate_yaw_rotation(landmarks, image_width, image_height):
    left_eye_inner = landmarks[133]  # Inner left eye
    right_eye_inner = landmarks[362]  # Inner right eye
    nose_tip = landmarks[1]  # Nose tip

    # Convert to image coordinates
    left_eye_inner_coords = np.array([left_eye_inner.x * image_width, left_eye_inner.y * image_height])
    right_eye_inner_coords = np.array([right_eye_inner.x * image_width, right_eye_inner.y * image_height])
    nose_tip_coords = np.array([nose_tip.x * image_width, nose_tip.y * image_height])

    # Calculate the horizontal distance between the eyes
    eye_distance = np.linalg.norm(left_eye_inner_coords - right_eye_inner_coords)

    # Calculate the horizontal distance between the nose and the midpoint of the eyes
    eye_midpoint = (left_eye_inner_coords + right_eye_inner_coords) / 2
    nose_eye_offset = nose_tip_coords[0] - eye_midpoint[0]  # Horizontal offset of the nose from the eye midpoint

    # Calculate yaw angle based on the ratio of the offset and the distance between the eyes
    yaw_angle = np.arctan2(nose_eye_offset, eye_distance) * (180 / np.pi)  # Convert to degrees

    return yaw_angle

# Function to align a single image based on the face landmarks
def align_image(image, target_eye_height=120, desired_eye_dist=70.0):
    initialize_face_mesh()  # Ensure that FaceMesh is initialized

    # Perform face landmark detection using MediaPipe
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            image_height, image_width, _ = image.shape
            landmarks = face_landmarks.landmark

            # Extract keypoints for left eye, right eye
            left_eye = landmarks[468]  # Center left eye
            right_eye = landmarks[473]  # Center right eye

            # Convert to image coordinates
            left_eye_coords = np.array([left_eye.x * image_width, left_eye.y * image_height])
            right_eye_coords = np.array([right_eye.x * image_width, right_eye.y * image_height])

            # Compute the angle between the eyes
            dx = right_eye_coords[0] - left_eye_coords[0]
            dy = right_eye_coords[1] - left_eye_coords[1]
            angle = math.atan2(dy, dx) * 180.0 / math.pi

            # Compute the current distance between the eyes
            current_eye_dist = math.sqrt((dx ** 2) + (dy ** 2))

            # Compute the scale factor to make the eye distance desired_eye_dist
            scale = desired_eye_dist / current_eye_dist

            # Compute the center between the two eyes
            eyes_center = ((left_eye_coords[0] + right_eye_coords[0]) // 2, 
                           (left_eye_coords[1] + right_eye_coords[1]) // 2)

            # Get the rotation matrix to rotate the image around the eyes center
            rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

            # Adjust the translation part of the rotation matrix to set the eyes at target_eye_height
            translation_x = rotation_matrix[0, 2]
            translation_y = rotation_matrix[1, 2] + (target_eye_height - eyes_center[1]) * scale

            rotation_matrix[0, 2] = translation_x
            rotation_matrix[1, 2] = translation_y

            # Compute the size of the new image after rotation and scaling
            output_size = (256, 256)

            # Apply the affine transformation (rotation + scaling + translation) to aligned images
            aligned_image = cv2.warpAffine(image, rotation_matrix, output_size)

            return aligned_image  # Return the aligned image for further processing
    else:
        print("No face detected in the image.")
        return None

# Function to process an image, align it, and then calculate the yaw angle based on the aligned image
def process_image(image, image_name, output_dir, target_eye_height=120, desired_eye_dist=70.0):
    # Step 1: Align the image
    aligned_image = align_image(image)

    if aligned_image is not None:

        # Perform face landmark detection again on the aligned image
        results = face_mesh.process(aligned_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                image_height, image_width, _ = aligned_image.shape
                landmarks = face_landmarks.landmark

                # Calculate the yaw angle on the aligned image
                yaw_angle = calculate_yaw_rotation(landmarks, image_width, image_height)

                if abs(yaw_angle) <= 20:
                    # Save the aligned image to the specified output directory
                    save_image(image_name, aligned_image, os.path.join(output_dir, "aligned"))
                    return aligned_image  # Return the aligned image for further processing
                else:
                    print(f"Image '{image_name}' has too much yaw (not a frontal face). Yaw: {yaw_angle}")
                    return None
        else:
            print(f"No face detected in the aligned image '{image_name}'.")
            return None
    else:
        print(f"Alignment failed for image '{image_name}'.")
        return None

# Function to load an image using OpenCV
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    return img

# Function to save an image using OpenCV
def save_image(image_path, image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image_bgr)


# Function to generate a noise image of the same size as the source image
def generate_noise_image(size):
    noise_r = np.array(Image.effect_noise(size, 35))
    noise_g = np.array(Image.effect_noise(size, 35))
    noise_b = np.array(Image.effect_noise(size, 35))
    noise = np.stack((noise_r, noise_g, noise_b), axis=2)
    noise_img = Image.fromarray(noise, mode='RGB')
    return noise_img


# Function to create custom damage on the source image
def create_custom_damage(img):
    height, width = img.shape[:2]
    mask_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_image)

    line_count_range = (30, 30)
    line_width_range = (8, 13)
    line_size_range_x = (10, 20)
    line_size_range_y = (10, 20)

    # Randomly select a point from within the defined area
    x = random.randint(90, 165)
    y = random.randint(100, 180)
    start_point = (x, y)

    # Draw a square at the starting point
    square_size = 20
    square_coords = (
        start_point[0] - square_size / 2, start_point[1] - square_size / 2, start_point[0] + square_size / 2,
        start_point[1] + square_size / 2)
    draw.rectangle(square_coords, fill=(0, 0, 0, 255))

    # Generate random lines on the image to simulate damage
    line_count = random.randint(*line_count_range)
    do_not_use = None

    for i in range(line_count):
        directions = [0, 1, 2, 3]
        if do_not_use:
            directions.remove(do_not_use)
        direction = random.choice(directions)

        if direction == 0:
            end_x = min(start_point[0] + random.randint(*line_size_range_x), width - 35)
            end_y = min(start_point[1] + random.randint(*line_size_range_y), height - 40)
            do_not_use = 1
        elif direction == 1:
            end_x = max(start_point[0] - random.randint(*line_size_range_x), 35)
            end_y = max(start_point[1] - random.randint(*line_size_range_y), 50)
            do_not_use = 0
        elif direction == 2:
            end_x = min(start_point[0] + random.randint(*line_size_range_x), width - 35)
            end_y = max(start_point[1] - random.randint(*line_size_range_y), 50)
            do_not_use = 3
        else:
            end_x = max(start_point[0] - random.randint(*line_size_range_x), 35)
            end_y = min(start_point[1] + random.randint(*line_size_range_y), height - 40)
            do_not_use = 2

        end_point = (end_x, end_y)
        draw.line((start_point, end_point), width=random.randint(*line_width_range), fill=(0, 0, 0, 255))
        start_point = end_point

    return mask_image


# Function to apply a mask to the image, replacing masked areas with noise
def apply_mask_to_image(source_img, mask_img):
    noise_img = generate_noise_image(source_img.shape[:2])
    # mask_img = mask_img.convert('L')  # Convert to grayscale
    damaged_img = Image.composite(noise_img, Image.fromarray(source_img), mask_img)
    return np.array(damaged_img)
