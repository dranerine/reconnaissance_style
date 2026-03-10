import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

img_height = 288
img_width = 432


# Transformations standards
transform = transforms.Compose(
    [
    transforms.Resize((128, 128)), 
    transforms.Grayscale(num_output_channels=1), # noir et blanc
    transforms.ToTensor(), # Convertit en [0, 1]
    transforms.Normalize((0.5,), (0.5,)) 
    ]
)

full_dataset = datasets.ImageFolder(root='Data/images_original', transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size


train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"Nombre d'images dans le dataset complet : {len(full_dataset)}")
print(f"Nombre d'images dans le dataset d'entraînement : {len(train_dataset)}")
print(f"Nombre d'images dans le dataset de test : {len(test_dataset)}")

print("Classes dans le dataset :", full_dataset.classes)


    
    
    
    