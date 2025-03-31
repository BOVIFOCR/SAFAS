import os, torch
from utils import protocol_decoder
import math
from .oulu_dataset import OuluFaceDataset
from .wfas_dataset import WFASFaceDataset
from .rose_dataset import RoseDataset
from .siw_dataset import SiWDataset


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_single_dataset(data_dir, FaceDataset, data_name="", train=True, label=None, img_size=256, map_size=32, transform=None, debug_subset_size=None, UUID=-1, exchange_aug=None, protocol_name=''):
    if train:
        if data_name in ["OULU"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'OULU-NPU/preposess'), split='train', label=label,
                                      transform=transform, UUID=UUID, exchange_aug=exchange_aug, protocol_name=protocol_name)
        elif data_name in ["CASIA_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'CASIA_faceAntisp/preposess'), split='train', label=label,
                                      transform=transform, UUID=UUID, exchange_aug=exchange_aug, protocol_name=protocol_name)
        elif data_name in ["Replay_attack"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'Replay/preposess'), split='train', label=label,
                                      transform=transform,  UUID=UUID, exchange_aug=exchange_aug, protocol_name=protocol_name)
        elif data_name in ["MSU_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'MSU-MFSD/preposess'), split='train', label=label,
                                      transform=transform,  UUID=UUID, exchange_aug=exchange_aug, protocol_name=protocol_name)
        elif data_name in ["ROSE_Youtu"]:
            data_set = RoseDataset(
                    data_name,
                    "/datasets1/rgpa18/datasets/rose_youtu_processed/",
                    is_train=True, label=label, transform=transform,
                    img_size=img_size)
        elif data_name in ["SiW"]:
            data_set = SiWDataset(
                    data_name,
                    "/datasets1/rgpa18/datasets/siw/",
                    is_trai=True, label=label, transform=transform,
                    img_size=img_size)

        if debug_subset_size is not None:
            data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    else:
        if data_name in ["OULU"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'OULU-NPU/preposess'), split='test', label=label,
                                      transform=transform, map_size=map_size, UUID=UUID)
        elif data_name in ["CASIA_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'CASIA_faceAntisp/preposess'), split='test', label=label,
                                      transform=transform, map_size=map_size, UUID=UUID)
        elif data_name in ["Replay_attack"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'Replay/preposess'), split='test', label=label,
                                      transform=transform, map_size=map_size, UUID=UUID)
        elif data_name in ["MSU_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'MSU-MFSD/preposess'), split='test', label=label,
                                      transform=transform, map_size=map_size, UUID=UUID)
        elif data_name in ["ROSE_Youtu"]:
            data_set = RoseDataset(
                    data_name,
                    "/datasets1/rgpa18/datasets/rose_youtu_processed/",
                    is_train=False, label=label, transform=transform,
                    map_size=map_size, UUID=UUID, img_size=img_size)
        elif data_name in ["SiW"]:
            data_set = SiWDataset(
                    data_name,
                    "/datasets1/rgpa18/datasets/siw/",
                    is_train=False, label=label, transform=transform,
                    map_size=map_size, UUID=UUID, img_size=img_size)

        if debug_subset_size is not None:
            data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    # print("Loading {}, number: {}".format(data_name, len(data_set)))
    return data_set

def get_oulu_split(data_dir, train=True, transform=None, debug_subset_size=None, UUID=-1):
    labels_file_name = ("TrainLMKE_jun_onlyspoofs" if train else "Test") + ".txt"
    data_set = OuluFaceDataset(data_dir, os.path.join(data_dir, "Protocols/Protocol_1/", labels_file_name), train, transform, UUID=UUID)
    if debug_subset_size is not None:
        data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    return data_set

def get_wfas_test(data_dir, transform=None, debug_subset_size=None, UUID=-1):
    data_set = WFASFaceDataset("/home/rgpa18/original_datasets/wfas/", transform, UUID=UUID)
    if debug_subset_size is not None:
        data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    return data_set

def get_protocol_name(protocol):
    return protocol.replace('_', '').replace('to', '2')

def get_datasets(data_dir, FaceDataset, train=True, protocol="1", img_size=256, map_size=32, transform=None, debug_subset_size=None, exchange_aug=None, test_on_wfas=False):

    if protocol == "O1":
        oulu = get_oulu_split(data_dir, train, transform, debug_subset_size, 0)
        print("Total number:", len(oulu))
        return oulu
    if test_on_wfas:
        wfas = get_wfas_test(data_dir, transform, debug_subset_size, 0)
        print("Total number:", len(wfas))
        return wfas

    data_name_list_train, data_name_list_test = protocol_decoder(protocol)

    protocol_name = get_protocol_name(protocol)
    sum_n = 0
    if train:
        data_set_sum = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[0], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=0, exchange_aug=exchange_aug, protocol_name=protocol_name)
        sum_n = len(data_set_sum)
        for i in range(1, len(data_name_list_train)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[i], train=True, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i, exchange_aug=exchange_aug, protocol_name=protocol_name)
            data_set_sum += data_tmp
            sum_n += len(data_tmp)
    else:
        data_set_sum = {}
        for i in range(len(data_name_list_test)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_test[i], train=False, img_size=img_size, map_size=map_size, transform=transform, debug_subset_size=debug_subset_size, UUID=i, exchange_aug=exchange_aug, protocol_name=protocol_name)
            data_set_sum[data_name_list_test[i]] = data_tmp
            sum_n += len(data_tmp)
    print("Total number: {}".format(sum_n))
    return data_set_sum
