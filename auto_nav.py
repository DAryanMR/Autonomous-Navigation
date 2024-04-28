import cv2
import numpy as np
import os
import math


def detect_objects(roi, obj, gray=False, threshold=0.8):
    result = cv2.matchTemplate(roi, obj, cv2.TM_CCORR_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > threshold:
        top_left = (max_loc[0] + roi_left, max_loc[1])
        bottom_right = (top_left[0] + obj.shape[1], top_left[1] + obj.shape[0])

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2) if not gray else cv2.rectangle(
            frame, top_left, bottom_right, (0, 0, 255), 2)

        return True, top_left, bottom_right

    return False, None, None


car_image = cv2.imread('my_car.png')

obstacles_folder = 'obstacles'
obstacle_images = []
for filename in os.listdir(obstacles_folder):
    if filename.endswith('.png'):
        obstacle_image = cv2.imread(os.path.join(
            obstacles_folder, filename), cv2.IMREAD_GRAYSCALE)
        obstacle_images.append(obstacle_image)


video_path = 'driving_footage.mp4'
video_capture = cv2.VideoCapture(video_path)

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(
    *'mp4v'), 30, (frame_width, frame_height))


roi_left = int(frame_width * 0.2)
roi_right = int(frame_width * 0.8)

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_rgb = frame[:, roi_left:roi_right]
    roi_gray = frame_gray[:, roi_left:roi_right]

    car_detected, car_top_left, car_bottom_right = detect_objects(
        roi_rgb, car_image, gray=False)

    if car_detected:

        car_center_x = int((car_top_left[0] + car_bottom_right[0]) / 2)
        car_center_y = int((car_top_left[1] + car_bottom_right[1]) / 2)
        free_road_left = True  # Assume free road on the left
        free_road_right = True  # Assume free road on the right

        if car_center_x-24 == roi_left:
            free_road_left = False  # No free road on the left if the car is in roi_left
        elif car_center_x+24 == roi_right:
            free_road_right = False  # No free road on the right if the car is in roi_right

        for obstacle_image in obstacle_images:
            obstacle_detected, obstacle_top_left, obstacle_bottom_right = detect_objects(roi_gray, obstacle_image,
                                                                                         gray=True)
            if obstacle_detected:
                obstacle_center_x = int(
                    (obstacle_top_left[0] + obstacle_bottom_right[0]) / 2)
                obstacle_center_y = int(
                    (obstacle_top_left[1] + obstacle_bottom_right[1]) / 2)

                distance = math.sqrt((car_center_x - obstacle_center_x) ** 2 + (
                    car_center_y - obstacle_center_y) ** 2)
                threshold = 400

                if distance <= threshold:
                    cv2.line(frame, (car_center_x, car_center_y), (obstacle_center_x, obstacle_center_y), (255, 0, 0),
                             2)
                    text = f"Distance: {distance:.2f}"
                    cv2.putText(frame, text, (obstacle_center_x, obstacle_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2, cv2.LINE_AA)
                    print("Collision with obstacle detected!")

                    if obstacle_center_y != car_center_y:
                        slope = (obstacle_center_x - car_center_x) / \
                            (obstacle_center_y - car_center_y)
                        if abs(slope) >= 0.5:
                            if obstacle_center_x < car_center_x:
                                free_road_left = False
                            else:
                                free_road_right = False

        if roi_left >= car_center_x:
            free_road_right = False
        if roi_right <= car_center_x:
            free_road_left = False

        cv2.putText(frame, "Free Road Left: {}".format(free_road_left), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Free Road Right: {}".format(free_road_right), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    output_video.write(frame)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
output_video.release()

cv2.destroyAllWindows()
