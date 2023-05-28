import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
from tqdm import tqdm

device = torch.device("mps")
threshold = 0.95

def validate(model, opt, data_loader, pre_model=None):
    with torch.no_grad():
        y_true, y_pred = [], []
        with tqdm(data_loader) as t:
            for img, label in t:
                # in_tens = img.cuda()
                in_tens = img.to(device)
                if pre_model:
                    in_tens = pre_model(in_tens).float()
                y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > threshold)
    acc = accuracy_score(y_true, y_pred > threshold)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
