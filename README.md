# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application using the MNIST dataset.


# Problem Statement and Dataset

Image denoising is the process of removing unwanted noise from images while preserving important features. In this experiment, a Convolutional Autoencoder is implemented using PyTorch to remove Gaussian noise from handwritten digit images.

The MNIST dataset is used, which contains grayscale handwritten digit images of size 28×28.


# DESIGN STEPS

## STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

## STEP 2:
Load the MNIST dataset and apply tensor transformation for preprocessing.

## STEP 3:
Add random Gaussian noise to the input images using a custom noise function.

## STEP 4:
Design the Convolutional Autoencoder with encoder and decoder layers using convolution and transpose convolution operations.

## STEP 5:
Train the model using Mean Squared Error (MSE) loss and Adam optimizer.

## STEP 6:
Test the trained model and visualize the Original, Noisy, and Reconstructed images.


## PROGRAM

### Name: Vikamuhan Reddy

### Register Number: 212223240181

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model
model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

summary(model, input_size=(1, 28, 28))


# Training function
def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for images, _ in loader:

            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(loader)}")


# Visualization function
def visualize_denoising(model, loader, num_images=10):

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)

            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    plt.figure(figsize=(18,6))

    for i in range(num_images):

        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```

### OUTPUT

### Model Summary
<img width="573" height="443" alt="image" src="https://github.com/user-attachments/assets/bcb1463e-1427-4d72-8e96-4d923fa96934" />


### Training loss
<img width="408" height="134" alt="image" src="https://github.com/user-attachments/assets/9e12227a-e6f9-4a53-90d9-3e8781d198fb" />


## Original vs Noisy Vs Reconstructed Image
<img width="1047" height="376" alt="image" src="https://github.com/user-attachments/assets/11a82d9a-529b-45d4-ad55-714fd27b40b9" />


## RESULT
Thus, the convolutional autoencoder for image denoising was successfully developed and trained using the MNIST dataset. The model effectively removed noise from handwritten digit images and reconstructed cleaner outputs.
