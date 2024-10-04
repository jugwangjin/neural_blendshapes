import os
import cv2

if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1]

    for d in os.listdir(input_dir):
        real_input_dir = os.path.join(input_dir, d)
        if not os.path.isdir(real_input_dir):
            continue
        # save vide on the  parent directory of input dir, with the same name
        output_dir = os.path.join(os.path.dirname(real_input_dir), os.path.basename(real_input_dir) + ".mp4")
        print(f"Saving video to {output_dir}")

        images = [img for img in sorted(os.listdir(real_input_dir)) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(real_input_dir, images[0]))

        # setting the frame width, height width
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_dir, fourcc, 30.0, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(real_input_dir, image)))

        cv2.destroyAllWindows()
        video.release()
        print("Video saved successfully")
    # Compare this snippet from flare/modules/geometry.py:
