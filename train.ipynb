import pandas as pd
import numpy as np
import cv2
import os
import re
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(10.0,10.0)
from google.colab import drive
drive.mount('/content/drive')
from IPython.display import display, clear_output
import seaborn as sns
from torchvision.ops import nms
#!pip install --upgrade gradio


#import gradio as gr
#image_input = gr.Image()
from torchvision.transforms import functional as F
from PIL import Image
#image_input = gr.inputs.Image()
#C:\Users\Maria\Downloads\Realwheatdetection.v1i.tensorflow\train
#INPUT_DIR = '/content/drive/My Drive/Realwheatdectection.v1i.tensorflow/train'
#TRAIN_DIR = os.path.join(INPUT_DIR, "train")
INPUT_DIR = '/content/drive/My Drive/wheatdetection.v1i.tensorflow/train'

# Load and Show Training Labels
df = pd.read_csv(os.path.join(INPUT_DIR, "_annotations.csv"))
print(df.head())

#function to test out seeing an image from the training set
def read_image_from_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found at path:", image_path)
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_image_from_train_folder(image_id):
    path = os.path.join(INPUT_DIR, image_id)
    return read_image_from_path(path)
sample_image_id = "3m_90_JPG.rf.9a4c224ea22a5032b8906ad726377c54.jpg"
image_array = (read_image_from_train_folder(sample_image_id))
if image_array is not None:
    plt.imshow(image_array)
    _ = plt.title(sample_image_id)
    plt.show()
else:
  print("Image failed to load")

def parse_bbox_text(string_input):
    input_without_brackets = re.sub("\[|\]", "", string_input)
    input_as_list = np.array(input_without_brackets.split(","))
    return input_as_list.astype(np.float32)
#top right bottom left cordinates
def xywh_to_x1y1x2y2(x,y,w,h):
    return np.array([x,y,x+w,y+h])
# Parse training bounding box labels
train_df = pd.read_csv(os.path.join(INPUT_DIR, "_annotations.csv"))
train_df['bbox'] = train_df[['xmin', 'ymin', 'xmax', 'ymax']].apply(lambda row: f"[{row['xmin']},{row['ymin']},{row['xmax']},{row['ymax']}]", axis=1)

# Now, you can apply the parse_bbox_text function on the 'bbox' column
# Ensure your parse_bbox_text function is correctly defined to handle the string format and convert it into a list or array
bbox_series = train_df['bbox'].apply(parse_bbox_text)

# Assuming parse_b
#bbox_series = train_df.bbox.apply(parse_bbox_text)

xywh_df = pd.DataFrame(bbox_series.to_list(), columns=["x", "y", "w", "h"])
xywh_df.reset_index(drop=True, inplace=True)

x2_df = pd.DataFrame(xywh_df.x + xywh_df.w, columns=["x2"]).reset_index(drop=True)
y2_df = pd.DataFrame(xywh_df.y + xywh_df.h, columns=["y2"]).reset_index(drop=True)

# Update training dataframe with parsed labels
train_df = pd.concat([train_df, xywh_df, x2_df, y2_df], axis=1)
train_df.head()
pd.set_option('display.max_columns', None)  # This line will ensure all columns are displayed
print(train_df.head())

#draw bounding boxes on the image
def draw_boxes_on_image(boxes, image, color=(255, 0, 0)):
    for box in boxes:
        x1, y1, x2, y2 = box  # Unpack the box values
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    return image
sample_image_id = train_df.filename.sample().item()
sample_image = read_image_from_train_folder(sample_image_id)
sample_bounding_boxes = train_df[train_df.filename == sample_image_id][["xmin", "ymin", "xmax", "ymax"]]
plt.imshow(draw_boxes_on_image(sample_bounding_boxes.to_numpy(), sample_image, color=(0, 200, 200)))
_ = plt.title(sample_image_id)
plt.show()
model = fasterrcnn_resnet50_fpn(pretrained=True)
#print(model)
#changing it to match our classes and data set
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=2)

# Verify the model architecture
model.roi_heads
#print(model.roi_heads)
#where to run the model so it doesnt take forever to run
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#move the images in batches to the device with an 80 / 20 split
def move_batch_to_device(images, targets):
    images= list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets
unique_image_ids = train_df['filename'].unique()
#n_validation = int(0.2 * len(unique_image_ids))
n_samples = 165
valid_ids =unique_image_ids[-n_samples:]
train_ids = unique_image_ids[:n_samples]
validation_df = train_df[train_df['filename'].isin(valid_ids)]
training_df = train_df[train_df['filename'].isin(train_ids)]
print("%i training samples\n%i validation samples" % (len(training_df.filename.unique()), len(validation_df.filename.unique())))
class WheatDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.image_ids = dataframe['filename'].unique()
        self.df = dataframe

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index:int):
            image_id = self.image_ids[index]
            image =read_image_from_train_folder(image_id).astype(np.float32)
            #scale image since pytorch expects images to be in the range of 0-1
            image /= 255.0
            #rearrange to match the expectations of the channels and convert it into a tensor
            image = torch.from_numpy(image).permute(2,0,1)
            records = self.df[self.df['filename'] == image_id]
            boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
            # Filter out boxes with zero width or height
            valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_boxes]
            #convert to tensor of float32
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            n_boxes = boxes.shape[0]
            labels = torch.ones((n_boxes,), dtype=torch.int64)
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            return image, target
train_dataset = WheatDataset(training_df)
valid_dataset = WheatDataset(validation_df)
def collate_fn(batch):
    return tuple(zip(*batch))
is_training_on_cpu = device == torch.device('cpu')
batch_size = 1 if is_training_on_cpu else 16
train_data_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
valid_data_loader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_fn)
batch_of_images, batch_of_targets = next(iter(train_data_loader))

sample_boxes = batch_of_targets[0]['boxes'].cpu().numpy().astype(np.int32)
sample_image = batch_of_images[0].permute(1,2,0).cpu().numpy() # convert b ack from pytorch format

plt.imshow(draw_boxes_on_image(sample_boxes, sample_image, color=(0,200,200)))
plt.show()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9)
checkpoint_path = '/content/drive/MyDrive/wheatdetection.v1i.tensorflow/fasterrcnngwhddd/fasterrcnngwhddd199.pth'  # Replace with your checkpoint path
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Add map_location argument

    #checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
else:
    start_epoch = 0
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9)
# *** Load checkpoint here ***

num_epochs = 25 if is_training_on_cpu else 200
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9)
train_losses = []
validation_losses = []
is_training = False
if is_training:
  model.train()

  for epoch in range(num_epochs):
      print("Epoch %i/%i " % (epoch + 1, num_epochs) )

      average_loss = 0
      for batch_id, (images, targets) in enumerate(train_data_loader):
          images, targets = move_batch_to_device(images, targets)
          loss_dict = model(images, targets)
          batch_loss = sum(loss for loss in loss_dict.values()) / len(loss_dict)
          optimizer.zero_grad()
          batch_loss.backward()
          optimizer.step()
          loss_value= batch_loss.item()
          average_loss = average_loss + (loss_value - average_loss) / (batch_id + 1)
          train_losses.append(average_loss)
          #clear_output(wait=True)

          print("Epoch %i/%i " % (epoch + 1, num_epochs))
          print("Mini-batch: %i/%i Loss: %.4f" % (batch_id + 1, len(train_data_loader), average_loss))

          # Optionally, print without clearing output every 100 mini-batches
          if batch_id % 100 == 0 and batch_id != 0:
            display(print("Checkpoint - Mini-batch: %i/%i Loss: %.4f" % (batch_id + 1, len(train_data_loader), average_loss)))
      model_save_dir ='/content/drive/MyDrive/wheatdetection.v1i.tensorflow/fasterrcnngwhddd'
      os.makedirs(model_save_dir, exist_ok=True)
      model_save_path = os.path.join(model_save_dir, f'fasterrcnngwhddd{epoch + 1}.pth')
      torch.save({
          'epoch': epoch,  # Save the current epoch number
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'training_losses': train_losses,
          'validation_losses': validation_losses
      }, model_save_path)
      print(f'Model saved at {epoch +1} to  {model_save_path}')
else:
  model.eval()
  predictions= []
  ground_truths = []
  iou_threshold = 0.6
  def make_validation_iter():
      valid_data_iter = iter(valid_data_loader)
      for images, targets in valid_data_iter:
          images, targets = move_batch_to_device(images, targets)

          cpu_device = torch.device("cpu")
          #outputs = model(images)
         # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
          with torch.no_grad():
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            #loss_dict = model(images, targets)
            '''
            if isinstance(loss_dict, dict):
                    batch_loss = sum(loss for loss in loss_dict.values()) / len(loss_dict)  # Calculate loss from dictionary values
            else:
                    # Handle the case where loss_dict is not a dictionary
                    batch_loss = torch.tensor(0.0)

            validation_losses.append(batch_loss.item())  # Ensure this line is here to append the batch_loss to validation_losses
'''

          for image, output, target in zip(images, outputs, targets):
              predicted_boxes = output['boxes'].cpu().detach().numpy().astype(np.int32)
              ground_truth_boxes = target['boxes'].cpu().numpy().astype(np.int32)
              image = image.permute(1,2,0).cpu().numpy()
              predictions.append(predicted_boxes)
              ground_truths.append(ground_truth_boxes)
              yield image, ground_truth_boxes, predicted_boxes

  validation_iter = make_validation_iter()
  #image, ground_truth_boxes, predicted_boxes = next(validation_iter)
  #image = draw_boxes_on_image(predicted_boxes, image, (255,0,0))
  #image = draw_boxes_on_image(ground_truth_boxes, image , (0,255,0))
  #plt.imshow(image)
  num_images_to_display = 8
  for _ in range(num_images_to_display):
      try:
          image, ground_truth_boxes, predicted_boxes = next(validation_iter)
          image_with_pred_boxes = draw_boxes_on_image(predicted_boxes, image.copy(), (255, 0, 0))
          image_with_gt_boxes = draw_boxes_on_image(ground_truth_boxes, image.copy(), (0, 255, 0))

          fig, axes = plt.subplots(1, 2, figsize=(15, 7))
          axes[0].imshow(image_with_pred_boxes)
          axes[0].set_title('Predicted Boxes')
          axes[1].imshow(image_with_gt_boxes)
          axes[1].set_title('Ground Truth Boxes')
          plt.show()
      except StopIteration:
          print("No more images in the validation set.")
          break
test_folder_path = '/content/drive/My Drive/realtester'

# List all files in the test folder
all_files = os.listdir(test_folder_path)

# Filter out files that are images (assuming .jpg here, adjust as needed)
image_files = [file for file in all_files if file.endswith('.JPG')]

# Optional: Create a DataFrame if you need structured data for further processing
test_images_df = pd.DataFrame(image_files, columns=['image_id'])

# Display the DataFrame to verify
print(test_images_df.head())
#INPUT_DIR = '/content/drive/My Drive/global-wheat-detection/test'
#TEST_DIR = os.path.join(INPUT_DIR, "test")
#test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

def read_image_from_test_folder(image_id):
    if not image_id.endswith('.JPG'):
        image_id += '.JPG'
    path = os.path.join(test_folder_path, image_id)
    # Proceed with loading t
    #path = os.path.join(test_folder_path, image_id + ".jpg")
    return read_image_from_path(path)
sample_image_id = "1.5"
image_array = (read_image_from_test_folder(sample_image_id))
if image_array is not None:
    plt.imshow(image_array)
    _ = plt.title(sample_image_id)
    plt.show()
else:
  print("Image failed to load")
class TestDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index:int):
        image_id = self.image_ids[index]
        # Assuming read_image_from_test_folder is a function you've defined to read images
        image = read_image_from_test_folder(image_id).astype(np.float32)
        # Scale image since PyTorch expects images to be in the range of 0-1
        image /= 255.0
        # Rearrange to match the expectations of the channels and convert it into a tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, image_id  # Return image and its ID instead of target


test_dataset = TestDataset(test_images_df)
def collate_fn(batch):
    return tuple(zip(*batch))
#is_training_on_cpu = device == torch.device('cpu')
#batch_size = 1 if is_training_on_cpu else 16
test_data_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)
model.eval()  # Set the model to evaluation mode
#checkpoint_path = '/content/drive/MyDrive/global-wheat-detection/fasterrcnngwhd/fasterrcnngwhd18.pth'
checkpoint_path = '/content/drive/MyDrive/wheatdetection.v1i.tensorflow/fasterrcnngwhddd/fasterrcnngwhddd199.pth'  # Replace with your checkpoint path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Determine device
checkpoint = torch.load(checkpoint_path, map_location=device)

from torchvision.ops import nms


def weighted_nms(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)

    indices = torch.arange(boxes.size(0))
    keep_boxes = []

    while indices.numel() > 0:
        max_score_index = scores[indices].argmax()
        max_score_box = boxes[indices[max_score_index]]
        keep_boxes.append(indices[max_score_index])

        if indices.numel() == 1:
            break

        remaining_indices = indices[indices != indices[max_score_index]]
        remaining_boxes = boxes[remaining_indices]

        ious = torchvision.ops.box_iou(max_score_box.unsqueeze(0), remaining_boxes).squeeze(0)
        overlapping_indices = remaining_indices[ious > iou_threshold]

        if overlapping_indices.numel() == 0:
            indices = remaining_indices
        else:
            overlapping_boxes = remaining_boxes[ious > iou_threshold]
            overlapping_scores = scores[overlapping_indices]

            combined_box = (
                max_score_box * scores[indices[max_score_index]] +
                (overlapping_boxes * overlapping_scores[:, None]).sum(dim=0)
            ) / (scores[indices[max_score_index]] + overlapping_scores.sum())

            boxes[indices[max_score_index]] = combined_box
            indices = remaining_indices[ious <= iou_threshold]

    return torch.tensor(keep_boxes, dtype=torch.int64)
def visualize_test_predictions(test_data_loader, model, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    cpu_device = torch.device("cpu")

    with torch.no_grad():
        for batch in test_data_loader:
            images, _ = batch
            images = list(image.to(device) for image in images)
            outputs = model(images)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            for image, output in zip(images, outputs):
                pred_boxes = output['boxes'].cpu().detach()
                scores = output['scores'].cpu().detach()

                # Filter out boxes with low scores
                high_score_indices = scores > score_threshold
                pred_boxes = pred_boxes[high_score_indices]
                scores = scores[high_score_indices]

                # Apply NMS to filter out overlapping boxes
                keep = weighted_nms(pred_boxes, scores, iou_threshold)
                filtered_boxes = pred_boxes[keep].numpy()

                image_np = image.permute(1, 2, 0).cpu().numpy()
                plt.imshow(draw_boxes_on_image(filtered_boxes, image_np, (255, 0, 0)))
                plt.title('Predicted Boxes')
                plt.show()

visualize_test_predictions(test_data_loader, model)
checkpoint_path = '/content/drive/MyDrive/wheatdetection.v1i.tensorflow/fasterrcnngwhddd/fasterrcnngwhddd199.pth'  # Replace with your checkpoint path
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Determine device
checkpoint = torch.load(checkpoint_path, map_location=device)
training_losses = checkpoint['training_losses']
validation_losses = checkpoint['validation_losses']

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training  Loss')
plt.legend()
plt.show()

model.load_state_dict(checkpoint['model_state_dict'])
def calculate_iou(box1, box2):
    x1_inter = np.maximum(box1[0], box2[0])
    y1_inter = np.maximum(box1[1], box2[1])
    x2_inter = np.minimum(box1[2], box2[2])
    y2_inter = np.minimum(box1[3], box2[3])

    inter_area = np.maximum(0, x2_inter - x1_inter + 1) * np.maximum(0, y2_inter - y1_inter + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou
def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        matched_gt_indices = set()  # Track which ground truth boxes have been matched

        # Evaluate predictions
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt_indices:
                TP += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                FP += 1

        # Calculate false negatives
        FN += len(gt_boxes) - len(matched_gt_indices)

    # Precision, recall, and F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

# Calculate metrics
precision, recall, f1_score = calculate_metrics(predictions, ground_truths)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
metrics = ['Precision', 'Recall', 'F1-score']
values = [precision, recall, f1_score]

# Create a bar chart
plt.bar(metrics, values)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.ylim([0, 1])  # Set y-axis limits to 0-1 for better visualization
plt.show()


