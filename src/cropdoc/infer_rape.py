import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
from torchvision import models
from torch import nn


def load_model(weights, classes):
    model = models.mobilenet_v3_small(pretrained=False)
    with open(classes) as f:
        class_names = json.load(f)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model, class_names


def main(img_path, weights, classes):
    model, class_names = load_model(weights, classes)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, 0)

    print(f"Pred: {class_names[top_idx]} ({top_prob:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--weights", default="runs_rapeseed/best.pt")
    parser.add_argument("--classes", default="runs_rapeseed/classes.json")
    args = parser.parse_args()

    main(args.img, args.weights, args.classes)