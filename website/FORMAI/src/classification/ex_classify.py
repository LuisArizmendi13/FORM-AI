import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.models.video import r3d_18
import torch.nn as nn

class ExerciseClassifier:
    def __init__(self, model_path=None, classes_path=None, fr=4, device=None):
        # Base directory where this file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Default to your new model filename
        model_path = model_path or os.path.join(base_dir, "models", "model_r3d18_3cat_wweights.pth")

        # Default classes file (assuming it's in the same directory as your script)
        classes_path = classes_path or os.path.join(base_dir, "classes.txt")

        self.fr = fr
        # Pick device (CPU or GPU if available)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load class labels from a text file
        self.classes = self._load_classes(classes_path)

        # Initialize the model (no pretrained weights, since we'll load our own state)
        self.model = r3d_18(weights=None)  
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

        # Load the saved state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Transformation to match training setup
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645],
                                 [0.22803, 0.22145, 0.216989]),
        ])

    def _load_classes(self, classes_path):
        """Load class labels from a text file (one per line, or custom format)."""
        with open(classes_path, "r") as f:
            return [line.strip().split(": ")[-1] for line in f.readlines()]

    def classify_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        clips = []
        frame_class = {}

        if not cap.isOpened():
            print("❌ Could not open video file.")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.fr == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tensor = self.transform(pil_img)
                clips.append(tensor)

                # Once we have 16 frames in our sliding window
                if len(clips) == 16:
                    clip = torch.stack(clips, dim=1).unsqueeze(0).to(self.device)  # [1, 3, T=16, H, W]
                    with torch.no_grad():
                        output = self.model(clip)
                        _, pred = torch.max(output, 1)
                        label = self.classes[pred.item()]

                    frame_class[label] = frame_class.get(label, 0) + 1
                    #print(label)

                    # Slide window by dropping the oldest frame
                    clips.pop(0)

            frame_idx += 1

        cap.release()

        if not frame_class:
            return None

        # Return the label that appeared most often
        return max(frame_class, key=frame_class.get)


if __name__ == "__main__":
    classifier = ExerciseClassifier()
    result = classifier.classify_video("dataset/examples/johncenasquat.mp4")
    print("Final classification:", result)


"""
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os

class ExerciseClassifier:
    def __init__(self, model_path=None, classes_path=None, fr=4, device=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(base_dir, "models/model.pth")
        classes_path = classes_path or os.path.join(base_dir, "classes.txt")

        self.fr = fr
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Load model
        from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
        self.model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self._load_num_classes(classes_path))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load class labels
        self.classes = self._load_classes(classes_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def _load_classes(self, classes_path):
        with open(classes_path, "r") as f:
            return [line.strip().split(": ")[-1] for line in f.readlines()]

    def _load_num_classes(self, classes_path):
        return sum(1 for _ in open(classes_path))

    def classify_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        frame_class = {}

        if not cap.isOpened():
            print("❌ Could not open video file.")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.fr == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    _, pred = torch.max(output, 1)
                    label = self.classes[pred.item()]

                if label not in frame_class:
                    frame_class[label] = 0
                frame_class[label] += 1 
                
                print(label)

            frame_idx += 1

        cap.release()

        if not frame_class:
            return None

        # Return the most frequent label
        return max(frame_class, key=frame_class.get) 
classifier = ExerciseClassifier()  
print(classifier.classify_video('dataset/examples/bench.mp4')) "
"""