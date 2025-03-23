from classification.ex_classify import ExerciseClassifier
from upload_image import upload_file_to_server
import os

classifier = ExerciseClassifier()

def run_classification(video_path):
    return classifier.classify_video(video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python integration.py <video_path>")
        exit(1)

    video_path = sys.argv[1]
    result = run_classification(video_path)

    # Absolute path to GIF (ensure it exists)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(base_dir, "earth.gif")

    url = upload_file_to_server(gif_path, "http://127.0.0.1:5000/upload")

    if url:
        print("✅ Uploaded to:", url)
    else:
        print("❌ Upload failed.")

    # Just print the classified exercise — DAIN will handle the prompting
    print(result)
