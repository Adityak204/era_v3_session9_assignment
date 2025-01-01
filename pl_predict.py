import torch
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import argparse


class ModelPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        # Initialize device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load the model
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Setup transforms - same as validation transforms from training
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load ImageNet class labels
        self.class_labels = self.load_imagenet_labels()

    def load_model(self, checkpoint_path: str):
        """Load the trained model from checkpoint"""
        from pl_train import ImageNetModule  # Import your module class

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model with same parameters as training
        model = ImageNetModule(
            learning_rate=0.156,  # Use the same parameters as in training
            batch_size=1,  # For inference, batch size doesn't matter
            num_workers=16,
            max_epochs=40,
            train_path="",  # These paths aren't needed for inference
            val_path="",
            checkpoint_dir="",
        )

        # Load state dict from checkpoint
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def load_imagenet_labels(self):
        """Load ImageNet class labels from JSON file"""
        # You'll need to download this file or create it
        # Format: {"0": "tench", "1": "goldfish", ...}
        labels_path = Path(
            "/Users/aditya/Documents/self_learning/ERA V3/week 9/imagenet-1k/data/imagenet-simple-labels.json"
        )

        if labels_path.exists():
            with open(labels_path) as f:
                data = json.load(f)
                data = {str(i + 1): name for i, name in enumerate(data)}
                return data
        return {str(i): f"class_{i}" for i in range(1000)}  # Fallback

    def predict_image(self, image_path: str, top_k: int = 5):
        """
        Make prediction for a single image

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            List of tuples (class_name, probability)
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)

            # Convert to list of (class_name, probability)
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.class_labels[str(idx.item())]
                predictions.append((class_name, prob.item()))

        return predictions


def main(checkpoint_path: str, image_path: str):
    # Example usage
    # checkpoint_path = (
    #     "/home/ec2-user/ebs/volumes/era_session9/resnet50-epoch18-acc53.7369.ckpt"
    # )
    predictor = ModelPredictor(checkpoint_path)

    # Make prediction
    # image_path = "/home/ec2-user/ebs/volumes/imagenet/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000019.JPEG"
    predictions = predictor.predict_image(image_path)

    # Print results
    print(f"\nPredictions for {image_path}:")
    for class_name, probability in predictions:
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    main(
        checkpoint_path=parser.parse_args().checkpoint,
        image_path=parser.parse_args().image,
    )
