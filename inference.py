import torch
import cv2
import os
import argparse
from torchvision.transforms import CenterCrop, Compose, Resize, Normalize, ToTensor
from dataset import PlantNetDataset
from model import build_model, model_params
from torch import nn


def get_args():
    parser = argparse.ArgumentParser(description="Inference CNN model")
    parser.add_argument("--image_path", "-a", type=str, default="test/a.jpg")
    parser.add_argument("--image_size", "-i", type=int, default=320)
    parser.add_argument("--checkpoint", "-p", type=str, default="trained_models/best.pt")
    parser.add_argument('--model_name', type=str, default="model_4")
    parser.add_argument('--num_classes', type=int, default=1081)
    args = parser.parse_args()
    return args


def inference(args):
    dataset = PlantNetDataset("plantnet_300K", "test", transform=None,
                              tail_augmentations=None)
    crop_size = model_params[args.model_name][2]

    model = build_model(args.model_name, args.num_classes)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_params"])
    else:
        print("No checkpoint provided")
        exit(0)


    transform = Compose([
        ToTensor(),
        Resize(size=args.image_size, antialias=True),
        CenterCrop(size=crop_size),
        Normalize(mean=[0.4425, 0.4695, 0.3266], std=[0.2353, 0.2219, 0.2325])
    ])

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)

    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
        prediction = softmax(prediction)
        conf_score, predicted_id = torch.max(prediction, dim=1)
    cv2.imshow(f"{dataset.get_class(predicted_id)} with score of {conf_score.item():.4f}", ori_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    inference(args)
