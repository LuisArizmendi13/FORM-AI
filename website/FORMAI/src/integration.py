from classification.ex_classify import ExerciseClassifier
from upload_image import upload_file_to_server
import os
import shutil

classifier = ExerciseClassifier()

def run_classification(video_path):
    return classifier.classify_video(video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python integration.py <video_path>")
        exit(1)

    video_path = sys.argv[1]
    ABS_PATH = os.path.abspath(__file__)
    assert 'website/FORMAI/src/integration.py' in ABS_PATH
    PROJECT_PATH = PROJECT_PATH[:ABS_PATH.rfind('website/FORMAI/src/integration.py')-1]

    # 1: go through pose demo on the input to get to get pose points

    for item in os.listdir(PROJECT_PATH + '/video_in'):
        item_path = os.path.join(PROJECT_PATH + 'video_in', item)
        if os.path.isfile(item_path): os.remove(item_path)
    for item in os.listdir(PROJECT_PATH + '/videopose3d_out'):
        item_path = os.path.join(PROJECT_PATH + 'videopose3d_out', item)
        if os.path.isfile(item_path): os.remove(item_path)
    shutil.copyfile(video_path, PROJECT_PATH + 'video_in')
    subprocess.run([PROJECT_PATH + '/pose_estimate_demo.py'])

    files = [item for item in os.listdir(PROJECT_PATH + '/videopose3d_out') if os.path.isfile(os.path.join(PROJECT_PATH + '/videopose3d_out', item))]
    assert len(files) == 1  # there should only be one file in the output dir
    test_pose_path = files[0]

    # 2: run the classification to figure out what exercise it is

    classification_result = run_classification(video_path)
    print(result) # Just print the classified exercise — DAIN will handle the prompting

    # 3: fetch the appropriate gold pose points for that exercise

    gold_pose_path = None
    if classification_result = 'deadlift': gold_pose_path = 'gold_deadlift.mp4.npy'
    elif classification_result = 'bench press': gold_pose_path = 'goldbench.mp4.npy'
    elif classification_result = 'squat': gold_pose_path = 'goldsquat.mp4.npy'
    else: assert False  # the classifier did not return a valid class

    gold_pose_path = PROJECT_PATH + '/gold_points/' + gold_pose_path

    # 4: create the 2 skeletons gif

    subprocess.run('python', PROJECT_PATH + '/draw_2_skeleton.py', gold_pose_path, test_pose_path)
    skeletons_path = PROJECT_PATH + f'/plots/{os.path.basename(gold_pose_path)}-{os.path.basename(test_pose_path)}-skeletons.gif'

    # 5: create error graph and error text description

    error_prompt = subprocess.run('python', PROJECT_PATH + '/draw_error_graphs.py', '--gold', gold_pose_path, '--test', test_pose_path)
    graph_path = PROJECT_PATH + f'/plots/{os.path.basename(gold_pose_path)}-{os.path.basename(test_pose_path)}-error_report.png'
    print(error_prompt)

    # 6: format the text properly into one text output

    # Absolute path to GIF (ensure it exists)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gif_path = os.path.join(base_dir, "earth.gif")

    # url = upload_file_to_server(gif_path, "http://127.0.0.1:5000/upload")
    url2 = upload_file_to_server(skeletons_path, "http://127.0.0.1:5000/upload")
    url3 = upload_file_to_server(graph_path, "http://127.0.0.1:5000/upload")

    if url2:
        print("✅ Uploaded to:", url2)
    else:
        print("❌ Upload failed.")

    if url3:
        print("✅ Uploaded to:", url3)
    else:
        print("❌ Upload failed.")
