from classification.ex_classify import ExerciseClassifier
from upload_image import upload_file_to_server
import os
import shutil
import subprocess

classifier = ExerciseClassifier()

def run_classification(video_path):
    return classifier.classify_video(video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python integration.py <video_path>")
        exit(1)

    video_path = sys.argv[1]

    # Proper project root
    ABS_PATH = os.path.abspath(__file__)
    PROJECT_PATH = os.path.abspath(os.path.join(ABS_PATH, "../../../../"))

    # 1: Run pose estimation
    video_in = os.path.join(PROJECT_PATH, "video_in")
    pose_out = os.path.join(PROJECT_PATH, "videopose3d_out")
    os.makedirs(video_in, exist_ok=True)
    os.makedirs(pose_out, exist_ok=True)

    for f in os.listdir(video_in):
        os.remove(os.path.join(video_in, f))
    for f in os.listdir(pose_out):
        os.remove(os.path.join(pose_out, f))

    shutil.copyfile(video_path, os.path.join(video_in, "input.mp4"))
    subprocess.run(["python3", "pose_estimate_demo.py"], cwd=PROJECT_PATH)

    output_files = os.listdir(pose_out)
    assert len(output_files) == 1
    test_pose_path = os.path.join(pose_out, output_files[0])

    # 2: Classify exercise
    classification_result = run_classification(video_path)
    print(classification_result)

    # 3: Match gold standard
    gold_map = {
        "deadlift": "gold_deadlift.mp4.npy",
        "bench press": "goldbench.mp4.npy",
        "squat": "goldsquat.mp4.npy"
    }
    gold_filename = gold_map.get(classification_result.lower())
    assert gold_filename, "Unknown exercise class"

    gold_pose_path = os.path.join(PROJECT_PATH, "gold_points", gold_filename)

    # 4: Generate skeleton GIF
    subprocess.run(["python3", "draw_2_skeleton.py", gold_pose_path, test_pose_path], cwd=PROJECT_PATH)
    skeletons_path = os.path.join(
        PROJECT_PATH, "plots", f"{os.path.basename(gold_pose_path)}-{os.path.basename(test_pose_path)}-skeletons.gif"
    )

    # 5: Generate error graph and prompt
    result = subprocess.run(
        ["python3", "draw_error_graphs.py", "--gold", gold_pose_path, "--test", test_pose_path],
        cwd=PROJECT_PATH,
        capture_output=True,
        text=True
    ) 
    print(result.stderr)
    error_prompt = result.stdout.strip()
    graph_path = os.path.join(
        PROJECT_PATH, "plots", f"{os.path.basename(gold_pose_path)}-{os.path.basename(test_pose_path)}-error_report.png"
    )

    # 6: Upload results
    url2 = upload_file_to_server(skeletons_path, "http://127.0.0.1:5000/upload")
    url3 = upload_file_to_server(graph_path, "http://127.0.0.1:5000/upload")

    print(url2 or "UPLOAD_FAILED")
    print(url3 or "UPLOAD_FAILED")
    print(error_prompt)
    print(classification_result)
