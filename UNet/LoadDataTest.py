
import torch
import numpy
from torch.utils.data import Dataset, DataLoader

numpy.random.seed(234)
torch.manual_seed(1234)


file_list_test = 'file_list_test.txt'
dataset_dir = 'D:\\NMDataset\\testset'
num_voxels = 128*128*128
factor_dose = 1.0e8
CT_dim = [128,128,128]
act_dose_coeff_511 = 0.026052890606224538
act_dose_coeff_140 = 0.0360363913971124 

class CTDoseDataset(Dataset):
    def __init__(self, data_aug=False, is_train=False, energy=511):
        self.data_aug = data_aug
        self.is_train = is_train
        self.energy = energy
        if energy == 140:
            self.act_dose_coeff = act_dose_coeff_140
        else:
            self.act_dose_coeff = act_dose_coeff_511
        self.ReadFileList()
        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def ReadFileList(self):

        file_list_fname = file_list_test
        with open(file_list_fname, 'r') as f:
            content = f.read()
        self.file_list = content.split('\n')

    def loadOneData(self, data_name):
        den_file = dataset_dir + "/" + data_name + "/density.raw"
        act_file = dataset_dir + "/" + data_name + "/act.raw"
        #dose_file = dataset_dir + "/" + data_name + "/NMDoseGPU_dose.raw"
        dose_file = dataset_dir + "/" + data_name + f"/dose{self.energy}.raw"
        density = numpy.fromfile(den_file, dtype=numpy.float32, count=num_voxels)
        act = numpy.fromfile(act_file, dtype=numpy.double, count=num_voxels)
        dose = numpy.fromfile(dose_file, dtype=numpy.float32, count=num_voxels)


        density = numpy.reshape(density, CT_dim)
        density = torch.from_numpy(density)

        max_act = numpy.max(act)
        sum_act = numpy.sum(act)
        act_norm = max_act/sum_act
        eval_max_dose = act_norm*self.act_dose_coeff
        act /= max_act

        act = (numpy.reshape(act, CT_dim)).astype(dtype=numpy.float32)
        act = torch.from_numpy(act)

        dose /= eval_max_dose
        dose = numpy.reshape(dose, CT_dim)
        dose = torch.from_numpy(dose)
        if self.data_aug:
            order = [0, 1, 2]
            numpy.random.shuffle(order)
            #print(order)
            dose = torch.permute(dose, order)
            act = torch.permute(act, order)
            density.permute(order)

        dose = torch.unsqueeze(dose,dim=0)
        data = torch.stack([density, act], axis=0)
        #data = torch.permute(data, [1,2,3,0])
        return data, dose

    def __getitem__(self, item):
        data_name = self.file_list[item]
        data, dose = self.loadOneData(data_name)

        return data, dose



if __name__ == "__main__":
    dataset = CTDoseDataset(data_aug=True)
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)

    for idx, data in enumerate(dataloader):
        X_train, Y_train = data
        print(X_train.shape)
        print(Y_train.shape)
        if idx > 2:
            break