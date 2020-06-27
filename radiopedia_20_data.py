import os
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.transforms import \
    Compose, AddChannel, LoadNifti, \
    ScaleIntensity, RandSpatialCrop, \
    ToTensor, CastToType, SpatialPad

monai.config.print_config()


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)
    
class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
    
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp




# Main function
def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Supervised learning data for training and validation
    data_dir = '/home/marafath/scratch/3d_seg_ct'
    fold = 5
    epc = 100
    
    for i in range(0,fold):
        train_images = []
        train_labels = []

        val_images = []
        val_labels = []
        for case in os.listdir(data_dir): 
            if i == 0:
                if int(case[2:4]) > 16:
                    train_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    train_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
                else:
                    val_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    val_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
            elif i == 1:
                if int(case[2:4]) < 17 or int(case[2:4]) > 32:
                    train_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    train_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
                else:
                    val_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    val_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
            elif i == 2:
                if int(case[2:4]) < 33 or int(case[2:4]) > 48:
                    train_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    train_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
                else:
                    val_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    val_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
            elif i == 3:
                if int(case[2:4]) < 49 or int(case[2:4]) > 64:
                    train_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    train_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
                else:
                    val_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    val_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
            elif i == 4:
                if int(case[2:4]) < 65:
                    train_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    train_labels.append(os.path.join(data_dir,case,'label.nii.gz'))
                else:
                    val_images.append(os.path.join(data_dir,case,'image.nii.gz'))
                    val_labels.append(os.path.join(data_dir,case,'label.nii.gz'))

        # Defining Transform
        train_imtrans = Compose([
            ScaleIntensity(),
            AddChannel(),
            CastToType(), #default is `np.float32`
            RandSpatialCrop((96, 96, 96), random_size=False),
            SpatialPad((96, 96, 96), mode='constant'),
            ToTensor()
        ])
        train_segtrans = Compose([
            AddChannel(),
            CastToType(), #default is `np.float32`
            RandSpatialCrop((96, 96, 96), random_size=False),
            SpatialPad((96, 96, 96), mode='constant'),
            ToTensor()
        ])
        val_imtrans = Compose([
            ScaleIntensity(),
            AddChannel(),
            CastToType(),
            SpatialPad((96, 96, 96), mode='constant'),
            ToTensor()
        ])
        val_segtrans = Compose([
            AddChannel(),
            CastToType(),
            SpatialPad((96, 96, 96), mode='constant'),
            ToTensor()
        ])

        # create a training data loader
        train_ds = NiftiDataset(train_images, train_labels, transform=train_imtrans, seg_transform=train_segtrans)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

        # create a validation data loader
        val_ds = NiftiDataset(val_images, val_labels, transform=val_imtrans, seg_transform=val_segtrans)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

        # Defining model and hyperparameters
        device = torch.device('cuda:0')
        model = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        loss_function = DC_and_CE_loss({'smooth': 1e-5, 'do_bg': False}, {})
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        # start a typical PyTorch training
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()
        writer = SummaryWriter()
        for epoch in range(epc):
            print('-' * 10)
            print('epoch {}/{}'.format(epoch + 1, epc))
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                # print('{}/{}, train_loss: {:.4f}'.format(step, epoch_len, loss.item()))
                writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    metric_sum = 0.
                    metric_count = 0
                    val_images_ = None
                    val_labels_ = None
                    val_outputs = None
                    for val_data in val_loader:
                        val_images_, val_labels_ = val_data[0].to(device), val_data[1].to(device)
                        roi_size = (160, 160, 96)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images_, roi_size, sw_batch_size, model)
                        value = compute_meandice(y_pred=val_outputs, y=val_labels_, include_background=False,
                                         to_onehot_y=True, sigmoid=False, mutually_exclusive=True)
                        metric_count += len(value)
                        metric_sum += value.sum().item()
                    metric = metric_sum / metric_count
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(),'/home/marafath/scratch/saved_models/radiopedia_best_f{}.pth'.format(i))
                        print("saved new best metric model")
                    print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    plot_2d_or_3d_image(val_images_, epoch + 1, writer, index=0, tag='image')
                    plot_2d_or_3d_image(val_labels_, epoch + 1, writer, index=0, tag='label')
                    plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag='output')

        print('train completed, best_metric: {:.4f} at epoch: {}'.format(best_metric, best_metric_epoch))
        writer.close()

if __name__ == '__main__':
    main()