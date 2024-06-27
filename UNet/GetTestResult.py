
from LoadDataTest import CTDoseDataset
from Network import UNet3D
from torch.utils.data import Dataset, DataLoader

import torch
import numpy

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(2)

energy = 511

MyLoss = torch.nn.L1Loss(reduction='mean')
val_dataset = CTDoseDataset(data_aug=False, is_train=False, energy=energy)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
in_ch = 2
out_ch = 1
model = UNet3D(in_ch, out_ch)


ckpt_file = f'511/ckpt_299.tar'
checkpoint = torch.load(ckpt_file)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

model.eval()

val_loss = 0.0
with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
        if i<20:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += MyLoss(out, y).item()

            result_output = (torch.squeeze(out)).cpu().numpy()
            y_output = (torch.squeeze(y)).cpu().numpy()

            result_output.tofile(f"test511/output_{i}.raw")
            y_output.tofile(f"test140/y_{i}.raw")

val_loss /= len(val_loader)

print(val_loss)




