# CNN_Project
# Animal Image Classification using CNN in PyTorch

This project implements an image classification model using a Convolutional Neural Network (CNN) to classify images of **dogs**, **cats**, and **pandas**. The model is built using PyTorch and trained on a labeled dataset containing images of animals organized in separate folders.

## Dataset

- Source: [Kaggle Dataset - animal-image-dataset (dog, cat, and panda)](https://www.kaggle.com/datasets/samuelcortinhas/animal-image-dataset-dog-cat-and-panda)
- Format: Images organized into class folders:
## Project Workflow Summary

Step 1: Extracted and merged datasets from .zip files.
Step 2: Verified image count and distribution across classes.
Step 3: Applied image transforms and created DataLoaders.
Step 4: Built CNN architecture and trained the model.
Step 5: Evaluated model performance and exported results.


## Model Architecture

A simple CNN was implemented with the following layers:
- 2 convolutional blocks (Conv2d → ReLU → MaxPool)
- Fully connected layers
- Dropout for regularization
- Output layer with softmax activation

```python
class AnimalCNN(nn.Module):
  def __init__(self):
      super(AnimalCNN, self).__init__()
      self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
      self.fc1 = nn.Linear(64 * 56 * 56, 128)
      self.dropout = nn.Dropout(0.5)
      self.fc2 = nn.Linear(128, 3)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 64 * 56 * 56)
      x = self.dropout(F.relu(self.fc1(x)))
      x = self.fc2(x)
      return x
```
## Training Results & Observations
- Model trained for 10 epochs using GPU in Google Colab.
- Achieved 99.92% training accuracy and 100% test accuracy.
- Loss decreased significantly indicating strong model fit.
- Used batch size 32 with data loaders.
## Evaluation metrics
- Test Accuracy: 99.92%.
- Confusion Matrix showed perfect classification.
## Challenges Faced & Solutions
- Challenge: Overfitting signs due to perfect accuracy.
- Solution: Plan to introduce data augmentation and dropout layers.
## Performance Graphs
- Plot 1: Accuracy vs. Epochs — to show both training and validation curves
- Plot 2: Loss vs. Epochs — to show stable convergence in loss
  
#### plot1
<img width="546" height="455" alt="image" src="https://github.com/user-attachments/assets/fcff8841-0aa1-42ed-969a-97ddf745f1c5" />

#### plot2
<img width="546" height="450" alt="image" src="https://github.com/user-attachments/assets/60bdc20e-a6f4-41b8-b212-3391238a92c1" />


## Conclusion
 This project demonstrates how a CNN can be trained on a small image dataset using PyTorch. It walks through data loading, model building, training, and evaluating on sample inputs. 
