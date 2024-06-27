
from LoadData import CTDoseDataset
from Network import UNet3D
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import torch
from torch.optim import Adam
from timeit import default_timer
from tqdm import tqdm


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


torch.manual_seed(0)
torch.set_num_threads(2)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#dataset
energy = 140
train_dataset = CTDoseDataset(data_aug=True, is_train=True, energy=energy)
val_dataset = CTDoseDataset(data_aug=False, is_train=False, energy=energy)

train_loader = DataLoader(train_dataset, batch_size=4,shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# network and loss function
in_channels = 2
out_channels = 1
model = UNet3D(in_channels=in_channels, out_channels=out_channels)
Net_num = sum(x.numel() for x in model.parameters())
print("Number of parameters:", Net_num)
model.to(device)

# optim
epochs = 300
learning_rate = 0.002
scheduler_step = 100
scheduler_gamma = 0.5

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

MyLoss = torch.nn.L1Loss(reduction='mean')
# train
for epoch in tqdm(range(epochs)):
    model.train()

    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = MyLoss(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    if (epoch+1) % 50 == 0 and epoch > 1:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        save_checkpoint(checkpoint, f"ckpt_{epoch}.tar")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = MyLoss(out, y)
            val_loss += mse.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
 


    print(epoch, train_loss, val_loss)