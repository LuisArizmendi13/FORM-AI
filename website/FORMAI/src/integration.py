from classification.ex_classify import ExerciseClassifier

classifier = ExerciseClassifier()

def run_classification(video_path):
    return classifier.classify_video(video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python integration.py <video_path>")
        exit(1)
    result = run_classification(sys.argv[1])
    print(result)
