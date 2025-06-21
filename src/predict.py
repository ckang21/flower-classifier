# src/predict.py
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import Flowers102
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Manually defined flower names list
flower_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot",
    "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily", "common tulip", "wild rose"
]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test data
test_data = Flowers102(root="data", split="test", download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Load model
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load("outputs/flower_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# Predict one image
# Predict one image (before the accuracy loop)
single_images, single_labels = next(iter(test_loader))
image = single_images[0].to(device).unsqueeze(0)
label = single_labels[0].item()

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()

# Evaluate accuracy on the whole test set
correct = 0
total = 0
test_loader_full = DataLoader(test_data, batch_size=1, shuffle=False)

for images, labels in test_loader_full:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# Show single image prediction
plt.imshow(single_images[0].cpu().permute(1, 2, 0))
plt.title(f"Actual: {flower_names[label]}\nPredicted: {flower_names[predicted_class]}")
plt.axis('off')
plt.show()
