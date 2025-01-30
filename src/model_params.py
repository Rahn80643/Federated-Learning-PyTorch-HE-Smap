import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchinfo import summary
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.transforms as T
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import logging.handlers
from time import time
from datetime import timedelta

# Settings for dataset and directories for logs
TRAIN_DATA_PATH = "/home/renyi/Data_Analytics/final_project/topic2/train/"
TEST_DATA_PATH = "/home/renyi/Data_Analytics/final_project/topic2/test/"
LOG_DIR = "./logs/"
MODEL_PATH = "./save/"
device = torch.device('cuda')

# Settings of hyperparamters
trainBatchSize = 16 
numEpochs = 1 # 100 #100 # 50
AdamDLR = 0.0001 


def eval_accuracy(all_labels, all_predictions, class_names):
    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    acc = (np.array(all_labels) == np.array(all_predictions)).sum()/ len(all_predictions)

    precision_per_class = precision_score(all_labels, all_predictions, average=None)
    recall_per_class = recall_score(all_labels, all_predictions, average=None)

    # confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    return precision, recall, acc, precision_per_class, recall_per_class, cm


def train_model(train_loader, validation_loader, model, numEpochs, logger, save_path):
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=AdamDLR)
    loss = nn.CrossEntropyLoss()  

    training_losses = []
    validation_losses = []

    train_time_start = time()
    for epoch in range(numEpochs):
        # train each batch in this epoch
        training_loss = 0.0
        for i, (batch_images, batch_labels) in enumerate(train_loader):
                
            X = batch_images.cuda()
            Y = batch_labels.cuda()
            preictions = model(X) 
            cost = loss(preictions, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            training_loss += cost.item()

            if (i+1) % 5 == 0:
                print('Epoch [%d/%d], lter %d, Loss: %.4f' %(epoch+1, numEpochs, i+1, cost.item()))
                logger.info('Epoch [%d/%d], lter %d, Loss: %.4f' %(epoch+1, numEpochs, i+1, cost.item()))

        # validate 
        model.eval()      
        # Disable gradient computation and reduce memory consumption.
        validation_cost_total = 0.0
        with torch.no_grad():
            for i, (validation_images, validation_labels) in enumerate(validation_loader):
                X = validation_images.cuda()
                Y = validation_labels.cuda()

                predictions = model(X)
                validation_cost = loss(predictions, Y).item()

                validation_cost_total += validation_cost
            
        avg_training_loss = training_loss / len(train_loader)
        avg_validation_cost = validation_cost_total/ len(validation_loader)

        training_losses.append(avg_training_loss)
        validation_losses.append(avg_validation_cost)
        
        print(f"\tAverage training loss: {avg_training_loss}, average validation loss: {avg_validation_cost}")
        logger.info(f"\tAverage training loss: {avg_training_loss}, average validation loss: {avg_validation_cost}")

        if (epoch % 5) == 0: # save model per 5 epochs
            torch.save(model, os.path.join(save_path, target_model+'_'+weight_setting+'_'+ str(epoch) + '.pth'))

    train_time_end = time()
    timedelta(seconds=int(train_time_end - train_time_start))
    print(f"Training takes {timedelta(seconds=(train_time_end - train_time_start))} seconds with {numEpochs} epochs.")
    logger.info(f"Training takes {timedelta(seconds=(train_time_end - train_time_start))} seconds with {numEpochs} epochs.")

    torch.save(model, os.path.join(save_path, target_model+'_'+weight_setting+'_final.pth'))

    return model, training_losses, validation_losses


# pretrained: load pretrained model and fine tune all parameters, fine_tuning: load pretrained model and fine tune only top layer 
def train_and_test(target_model, weight_setting): 
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    
    if not os.path.exists(os.path.join(MODEL_PATH, target_model)):
        os.mkdir(os.path.join(MODEL_PATH, target_model))

    # Create a logger
    logger = logging.getLogger(target_model+weight_setting)
    logger.setLevel(logging.INFO)  # Set the log level to INFO
    file_handler = logging.FileHandler(os.path.join(MODEL_PATH, target_model, target_model+'_'+weight_setting+'_log.txt'))
    file_handler.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(file_handler)


    print(f"Train {target_model} with {weight_setting}, \
            Learning rate for optimizer: {AdamDLR}, \
            training batch size: {trainBatchSize}, \
            total epochs: {numEpochs}")

    logger.info(f"Train {target_model} with {weight_setting}, \
            Learning rate for optimizer: {AdamDLR}, \
            training batch size: {trainBatchSize}, \
            total epochs: {numEpochs}")

    if target_model == "alexNet":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.alexnet
        NNInputSize = 224 
        pixelMean = [0.485, 0.456, 0.406]
        pixelStd = [0.229, 0.224, 0.225]

    elif target_model == "resnet18":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
        NNInputSize = 224 
        pixelMean = [0.485, 0.456, 0.406]
        pixelStd = [0.229, 0.224, 0.225]

    elif target_model == "mobilenetv2":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.mobilenet_v2
        NNInputSize = 224 
        pixelMean = [0.485, 0.456, 0.406]
        pixelStd = [0.229, 0.224, 0.225]
    
    elif target_model == "mobilenetv3":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small
        NNInputSize = 224 
        pixelMean = [0.485, 0.456, 0.406]
        pixelStd = [0.229, 0.224, 0.225]

    
    elif target_model == "squeezenet1_1":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_0.html#torchvision.models.squeezenet1_0
        NNInputSize = 224 
        pixelMean = [0.485, 0.456, 0.406]
        pixelStd = [0.229, 0.224, 0.225]

    
    elif target_model == "vit":
        # https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16
        NNInputSize = 224 
        pixelMean = [0.485, 0.456, 0.406]
        pixelStd = [0.229, 0.224, 0.225]


    # Define transformations and data augmentation for the training data and testing data
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(NNInputSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(pixelMean, pixelStd)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(NNInputSize),
        transforms.CenterCrop(NNInputSize),
        transforms.ToTensor(),
        transforms.Normalize(pixelMean, pixelStd)
    ])

    # Load the datasets with ImageFolder
    full_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transforms)

    # Define the size of your validation set
    validation_size = int(0.2 * len(full_dataset))  # 20% falidation
    training_size = len(full_dataset) - validation_size # 80% for training

    # Split the dataset into training and validation datasets
    train_dataset, validation_dataset = random_split(full_dataset, [training_size, validation_size])

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=trainBatchSize, shuffle=True)
    # Create a DataLoader for the validation set
    validation_loader = DataLoader(validation_dataset, batch_size=trainBatchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Invert the class_to_idx dictionary to get a mapping from index to class
    class_names = {v: k for k, v in full_dataset.class_to_idx.items()}

    # Analyze dataset
    # Count the number of images per class
    class_counts = {}
    for _, label in full_dataset.samples:
        class_name = full_dataset.classes[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    # Display the counts
    print("Number of images per class:")
    logger.info("Number of images per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
        logger.info(f"{class_name}: {count}")

    # Check for imbalance
    total_images = sum(class_counts.values())
    max_images = max(class_counts.values())
    min_images = min(class_counts.values())

    print("\nDataset statistics:")
    print(f"Total number of images: {total_images}")
    print(f"Largest class has {max_images} images")
    print(f"Smallest class has {min_images} images")

    logger.info("\nDataset statistics:")
    logger.info(f"Total number of images: {total_images}")
    logger.info(f"Largest class has {max_images} images")
    logger.info(f"Smallest class has {min_images} images")

    # Transfer learning
    if target_model == "alexNet":
        if weight_setting == "pretrained":
            model = models.alexnet(pretrained=True).to(device)
        elif weight_setting == "scratch":
            model = models.alexnet(pretrained=False).to(device)
        elif weight_setting == "fine_tuning":
            model = models.alexnet(pretrained=True).to(device)
            # ConvNet as fixed feature extractor
            for param in model.parameters():
                param.requires_grad = False

        model.classifier  = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 19),
        )


    if target_model == "resnet18":
        if weight_setting == "pretrained":
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT').to(device)
        
        elif weight_setting == "scratch":
            model = models.resnet18().to(device)

        elif weight_setting == "fine_tuning":
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT').to(device)

            # ConvNet as fixed feature extractor
            for param in model.parameters():
                param.requires_grad = False
    
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=19)
    
    if target_model == "mobilenetv2":
        if weight_setting == "pretrained":
            model = models.mobilenet_v2(pretrained=True).to(device)
        
        elif weight_setting == "scratch":
            model = models.mobilenet_v2(pretrained=False).to(device)

        elif weight_setting == "fine_tuning":
            model = models.mobilenet_v2(pretrained=True).to(device)

            # ConvNet as fixed feature extractor
            for param in model.parameters():
                param.requires_grad = False
    
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=19)

    if target_model == "mobilenetv3":
        if weight_setting == "pretrained":
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1').to(device)
        
        elif weight_setting == "scratch":
            model = models.mobilenet_v3_small().to(device)

        elif weight_setting == "fine_tuning":
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1').to(device)

            # ConvNet as fixed feature extractor
            for param in model.parameters():
                param.requires_grad = False
    
        model.classifier[3] = torch.nn.Linear(in_features=model.classifier[3].in_features, out_features=19)


    elif target_model == "squeezenet1_1":
        if weight_setting == "pretrained":
            model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights).to(device)
        
        elif weight_setting == "scratch":
            model = models.squeezenet1_1().to(device)

        elif weight_setting == "fine_tuning":
            model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights).to(device)

            # ConvNet as fixed feature extractor
            for param in model.parameters():
                param.requires_grad = False


        model.classifier[1] = torch.nn.Conv2d(in_channels=model.classifier[1].in_channels, out_channels=19, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = 19
        stop = 1

    elif target_model == "vit":
        if weight_setting == "pretrained":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)

        elif weight_setting == "scratch":
            model = models.vit_b_16().to(device)

        elif weight_setting == "fine_tuning":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)      

            # ConvNet as fixed feature extractor
            for param in model.parameters():
                param.requires_grad = False          

        # change the number of output classes
        model.heads = nn.Linear(in_features=768, out_features=19, bias=True)


    save_path = os.path.join(MODEL_PATH, target_model)
    print(summary(model, (trainBatchSize, 3, NNInputSize, NNInputSize), depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=device))
    logger.info(summary(model, (trainBatchSize, 3, NNInputSize, NNInputSize), depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=device))
    trained_model, training_losses, validation_losses = train_model(train_loader, validation_loader, model, numEpochs=numEpochs, logger=logger, save_path=save_path)
    
    plt.style.use('seaborn-whitegrid')      
    plt.figure(figsize=(10, 7))  
    plt.plot(training_losses, 'b-', label='average training loss')
    plt.plot(validation_losses, 'r-', label='average validation loss')
    plt.title('Training and validation loss')
    plt.xlabel("#Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plot_name = f"{target_model}_{weight_setting}_loss_curve.png"
    plt.savefig(os.path.join(MODEL_PATH, target_model, plot_name))
    plt.close()
    
    trained_model.eval()

    # Initialize lists to store all predictions and true labels
    all_predictions = []
    all_labels = []

    # No need to track gradients here
    test_time_start = time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    test_time_end = time()

    print(f"testing {len(test_loader.dataset)} images takes {timedelta(seconds=(test_time_end - test_time_start))} seconds")
    logger.info(f"testing {len(test_loader.dataset)} images takes {timedelta(seconds=(test_time_end - test_time_start))} seconds")

    # Calculate accuracy results
    precision, recall, acc, precision_per_class, recall_per_class, cm = eval_accuracy(all_labels, all_predictions, class_names)
    
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}')
    logger.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}')
    
    # Print the scores for each class
    print("Scores per class:")
    logger.info("Scores per class:")
    for key in class_names:
        print(f"\t{class_names[key]}: Precision = {precision_per_class[key]:.2f}, "
              f"Recall = {recall_per_class[key]:.2f}")
        logger.info(f"\t{class_names[key]}: Precision = {precision_per_class[key]:.2f}, "
              f"Recall = {recall_per_class[key]:.2f}")
    

    # Plotting using seaborn for an enhanced visual representation
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(MODEL_PATH, target_model, target_model+'_'+weight_setting+'_confusion_matrix.png'))
    plt.close()

    torch.cuda.empty_cache()
    # plt.show()

def test_model(NNInputSize, load_path = None):  
    # After checking pixel normalization, the model architectures used in this project have the same pixel mean and std
    pixelMean = [0.485, 0.456, 0.406]
    pixelStd = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
        transforms.Resize(NNInputSize),
        transforms.CenterCrop(NNInputSize),
        transforms.ToTensor(),
        transforms.Normalize(pixelMean, pixelStd)
    ])

    # Load the datasets with ImageFolder
    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Invert the class_to_idx dictionary to get a mapping from index to class
    class_names = {v: k for k, v in test_dataset.class_to_idx.items()}
        
    trained_model = torch.load(load_path)
    
    trained_model.eval()

    # Initialize lists to store all predictions and true labels
    all_predictions = []
    all_labels = []

    # No need to track gradients here
    test_time_start = time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    test_time_end = time()

    print(f"testing {len(test_loader.dataset)} images takes {timedelta(seconds=(test_time_end - test_time_start))} seconds")

    # Calculate accuracy results
    precision, recall, acc, precision_per_class, recall_per_class, cm = eval_accuracy(all_labels, all_predictions, class_names)
    
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}')
        
    # Print the scores for each class
    print("Scores per class:")
    
    for key in class_names:
        print(f"\t{class_names[key]}: Precision = {precision_per_class[key]:.2f}, "
              f"Recall = {recall_per_class[key]:.2f}")
        
    

    # Plotting using seaborn for an enhanced visual representation
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.close()

    torch.cuda.empty_cache()
    plt.show()    


if __name__ == "__main__":

    resnet18_model = models.resnet18(weights='ResNet18_Weights.DEFAULT').to(device)
    resnet34_model = models.resnet34(weights='ResNet34_Weights.DEFAULT').to(device)
    resnet50_model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)

    from torchinfo import summary
    # Print experiment summary
    resnet18_result = str(summary(resnet18_model, (1, 3, 224, 224), depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device='cuda'))
    print('\n' + resnet18_result)

    resnet34_result = str(summary(resnet34_model, (1, 3, 224, 224), depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device='cuda'))
    print('\n' + resnet34_result)

    resnet50_result = str(summary(resnet50_model, (1, 3, 224, 224), depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device='cuda'))
    print('\n' + resnet50_result)

    stop =1
    # target_models = ["alexNet", "resnet18", "mobilenetv2", "mobilenetv3", "vit", "squeezenet1_1"]
    
    # # pretrained: load pretrained model and fine tune all parameters, fine_tuning: load pretrained model and fine tune only top layer 
    # weight_settings = ["pretrained"]

    # for target_model in target_models:
    #     for weight_setting in weight_settings:
    #         print(f"Prepare to train and test {target_model} with weight set as {weight_setting}")
    #         train_and_test(target_model, weight_setting)


    # Set the path of model and uncomment to execute testing only
    # This link contains the models I trained: https://drive.google.com/drive/folders/1t2hnpZrHyGOZJ-th2VvzRPsYINkVIKqf?usp=sharing
    '''
    model_list = [
        "./save_100_epochs/alexNet_fine_tuning_final.pth",
        "./save_100_epochs/resnet18_fine_tuning_final.pth",
        "./save_100_epochs/mobilenetv2_fine_tuning_final.pth",
        "./save_100_epochs/mobilenetv3_fine_tuning_final.pth",
        "./save_100_epochs/squeezenet1_1_fine_tuning_final.pth",
        "./save_100_epochs/vit_fine_tuning_final.pth",

        "./save_100_epochs/alexNet_pretrained_final.pth",
        "./save_100_epochs/resnet18_pretrained_final.pth",
        "./save_100_epochs/mobilenetv2_pretrained_final.pth",
        "./save_100_epochs/mobilenetv3_pretrained_final.pth",
        "./save_100_epochs/squeezenet1_1_pretrained_final.pth",
        "./save_100_epochs/vit_pretrained_final.pth",

        "./save_100_epochs/alexNet_scratch_final.pth",
        "./save_100_epochs/resnet18_scratch_final.pth",
        "./save_100_epochs/mobilenetv2_scratch_final.pth",
        "./save_100_epochs/mobilenetv3_scratch_final.pth",
        "./save_100_epochs/squeezenet1_1_scratch_final.pth",
        "./save_100_epochs/vit_scratch_final.pth",

        "./save_scratch_500_epochs/alexNet_scratch_final.pth",
        "./save_scratch_500_epochs/resnet18_scratch_final.pth",
        "./save_scratch_500_epochs/mobilenetv2_scratch_final.pth",
        "./save_scratch_500_epochs/mobilenetv3_scratch_final.pth",
        "./save_scratch_500_epochs/squeezenet1_1_scratch_final.pth",
        "./save_scratch_500_epochs/vit_scratch_final.pth",

        "./save_scratch_1000_epochs/alexNet_scratch_final.pth",
        "./save_scratch_1000_epochs/resnet18_scratch_final.pth",
        "./save_scratch_1000_epochs/mobilenetv2_scratch_final.pth",
        "./save_scratch_1000_epochs/mobilenetv3_scratch_final.pth",
        "./save_scratch_1000_epochs/squeezenet1_1_scratch_final.pth",
        "./save_scratch_1000_epochs/vit_scratch_final.pth",
    ]

    for model in model_list:
        print(f"result of {model}")
        test_model(224, model)
        print("=========\n")
    '''