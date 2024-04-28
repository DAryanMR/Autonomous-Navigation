import cv2
import numpy as np
import os


def detect_objects(roi, obj, gray=False, threshold=0.8):
    # if exceptions:
    #     for exception in exceptions:
    #         result = cv2.matchTemplate(roi, exception, cv2.TM_CCOEFF_NORMED)
    #         _, max_val, _, _ = cv2.minMaxLoc(result)

    #         if max_val > threshold:
    #             return False

    result = cv2.matchTemplate(roi, obj, cv2.TM_CCORR_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > threshold:
        top_left = (max_loc[0] + roi_left, max_loc[1])
        bottom_right = (top_left[0] + obj.shape[1], top_left[1] + obj.shape[0])

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2) if not gray else cv2.rectangle(
            frame, top_left, bottom_right, (0, 0, 255), 2)

        return True


car_image = cv2.imread('my_car.png')

obstacles_folder = 'obstacles'
obstacle_images = []
for filename in os.listdir(obstacles_folder):
    if filename.endswith('.png'):
        obstacle_image = cv2.imread(os.path.join(
            obstacles_folder, filename), cv2.IMREAD_GRAYSCALE)
        obstacle_images.append(obstacle_image)

# exceptions_folder = 'exceptions'
# exception_images, exception_images_gray = [], []
# for filename in os.listdir(exceptions_folder):
#     if filename.endswith('.png'):
#         exception_image = cv2.imread(os.path.join(
#             exceptions_folder, filename))
#         exception_image_gray = cv2.imread(os.path.join(
#             exceptions_folder, filename), cv2.IMREAD_GRAYSCALE)
#         exception_images.append(exception_image)
#         exception_images_gray.append(exception_image_gray)

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

    car_detected = detect_objects(
        roi_rgb, car_image, gray=False)

    if car_detected:
        for obstacle_image in obstacle_images:
            detect_objects(roi_gray, obstacle_image, gray=True)

    output_video.write(frame)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
output_video.release()

cv2.destroyAllWindows()
