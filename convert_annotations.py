import os


def convert_to_yolo_format(image_width, image_height, points):
    """
    Converts a list of points to YOLO format bounding box (class_id, center_x, center_y, width, height)
    """
    x_coords = [points[i] for i in range(0, len(points), 2)]
    y_coords = [points[i] for i in range(1, len(points), 2)]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    center_x = (x_min + x_max) / 2 / image_width
    center_y = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return center_x, center_y, width, height


def process_file(input_path, output_path, image_width, image_height):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            class_id = parts[0]
            points = list(map(float, parts[1:]))
            center_x, center_y, width, height = convert_to_yolo_format(image_width, image_height, points)
            outfile.write(f"{class_id} {center_x} {center_y} {width} {height}\n")


def convert_annotations(input_dir, output_dir, image_width, image_height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_file(input_path, output_path, image_width, image_height)


if __name__ == "__main__":
    image_width = 1280  # Replace with your image width
    image_height = 720  # Replace with your image height
    input_dir = 'data/labels/train'  # Replace with your input annotations directory
    output_dir = 'data/labels/train_yolo'  # Replace with your output annotations directory

    convert_annotations(input_dir, output_dir, image_width, image_height)
