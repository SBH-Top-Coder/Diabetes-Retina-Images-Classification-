import torch
from torch import nn, optim
import os
import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction,
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    print('Train Begin ')
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        #save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.unsqueeze(1).float())

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss average over epoch: {sum(losses)/len(losses)}")


def main():
    # 1 . Preparing DataSets : 
    train_ds = DRDataset(
        images_folder="/home/semi/Image_Classification/Full_Preproceed/",
        path_to_csv="/home/semi/Image_Classification/Labels.csv",
        transform=Config.val_transforms,
    )
    test_ds = DRDataset(
        images_folder="/home/semi/Image_Classification/Full_Preproceed/",
        path_to_csv="/home/semi/Image_Classification/Labels.csv",
        transform=Config.val_transforms,
        # train=False,
    )
    # val_ds = DRDataset(
    #     images_folder="/home/semi/Image_Classification/Full_Preproceed/",
    #     path_to_csv="/home/semi/Image_Classification/Labels.csv",
    #     transform=Config.val_transforms,
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, num_workers=6, shuffle=False
    )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=Config.BATCH_SIZE,
    #     num_workers=2,
    #     pin_memory=Config.PIN_MEMORY,
    #     shuffle=False,
    # )
    loss_fn = nn.MSELoss()
    # 2. Building Model : 
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 1)
    model = model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if Config.LOAD_MODEL and Config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(Config.CHECKPOINT_FILE), model, optimizer, Config.LEARNING_RATE)

    make_prediction(model, train_loader, "submission_.csv")
    preds, labels = check_accuracy(train_loader, model, Config.DEVICE)
    # print ("The f1 score currently is " , f1_score(preds,labels) ) 
    print("predictions", preds)
    print("Labels " , labels)
    print ("The f1 score currently is " , f1_score(preds,labels,average = 'weighted')  ) 



    # for epoch in range(Config.NUM_EPOCHS):
    #     train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, Config.DEVICE)        
    #     # get on train
    #     preds, labels = check_accuracy(test_loader, model, Config.DEVICE)
    #     print ("The f1 score currently is " , f1_score(preds,labels,average = 'weighted') ) 

    #     if Config.SAVE_MODEL:
    #         checkpoint = {
    #             "state_dict": model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #         }
    #         save_checkpoint(checkpoint, filename=f"b3_{epoch}.pth.tar")



if __name__ == "__main__":
    main()