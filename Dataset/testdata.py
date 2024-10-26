from DatasetTaxi import TaxiDataset
import matplotlib.pyplot as plt
import torch

#

filepath = "D:\\DDPM\\testingcode\\TrajWeaver\\Dataset\\test_Xian_B100_l512_E05.pth"
taxidata = TaxiDataset(max_len=512, load_path=filepath)

print(len(taxidata.trajs))
for i in range(len(taxidata.trajs)):
    data = taxidata.trajs[i]
    print(data.shape)


filepath = "D:\\DDPM\\testingcode\\TrajWeaver\\Dataset\\test_Chengdu_B100_l512_E05.pth"
taxidata = TaxiDataset(max_len=512, load_path=filepath)

print(len(taxidata.trajs))
for i in range(len(taxidata.trajs)):
    data = taxidata.trajs[i]
    print(data.shape)




filepath = "D:\\DDPM\\testingcode\\TrajWeaver\\Dataset\\test_20240711_B100_l512_E05.pth"
taxidata = TaxiDataset(max_len=512, load_path=filepath)

print(len(taxidata.trajs))
for i in range(len(taxidata.trajs)):
    data = taxidata.trajs[i]
    print(data.shape)