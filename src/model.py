
import torch
import torch.nn as nn
from torchvision import models


class ChessSquareClassifier(nn.Module):
    def __init__(self, num_classes=13, pretrained=True, model_name='resnet50'):

        super(ChessSquareClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        print(f"Created {model_name} model:")
        print(f"  Pretrained: {pretrained}")
        print(f"  Num classes: {num_classes}")
        print(f"  Final layer: Linear({in_features} -> {num_classes})")
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x, return_probs=False):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        if return_probs:
            return predictions, probs
        return predictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == '__main__':
    print("=" * 60)
    print("TESTING ChessSquareClassifier")
    print("=" * 60)
    
    # Create model
    model = ChessSquareClassifier(num_classes=13, pretrained=True, model_name='resnet50')
    
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, 13]")
    
    # Test prediction
    print("\nTesting prediction...")
    preds, probs = model.predict(dummy_input, return_probs=True)
    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Max probability per sample: {probs.max(dim=1).values}")
    
    print("\n" + "=" * 60)
    print("Model test passed!")
    print("=" * 60)