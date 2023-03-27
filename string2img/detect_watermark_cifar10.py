import argparse
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution",
    type=int,
    help="Height and width of square images.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)

args = parser.parse_args()

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms


if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_decoder():
    global RevealNet
    global FINGERPRINT_SIZE

    from models import StegaStampDecoder
    state_dict = torch.load(args.decoder_path)
    FINGERPRINT_SIZE = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.image_resolution, 3, FINGERPRINT_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)


def load_data():
    global dataset, dataloader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def extract_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []
    bitwise_accuracy = 0

    BATCH_SIZE = args.batch_size
    
    # transform gt_fingerprints to 
    gt_fingerprints  = "0100"
    fingerprint_size = len(gt_fingerprints)
    z = torch.zeros((args.batch_size, fingerprint_size), dtype=torch.float)
    for (i, fp) in enumerate(gt_fingerprints):
        z[:, i] = int(fp)
    z = z.cuda()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for images, _ in tqdm(dataloader):
        images = images.to(device)

        fingerprints = RevealNet(images)
        fingerprints = (fingerprints > 0).long()

        bitwise_accuracy += (fingerprints[: images.size(0)].detach() == z[: images.size(0)]).float().mean(dim=1).sum().item()

        all_fingerprinted_images.append(images.detach().cpu())
        all_fingerprints.append(fingerprints.detach().cpu())

    dirname = args.output_dir
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
    print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}") # non-corrected
          
    # write in file
    # f = open(os.path.join(args.output_dir, "detected_fingerprints.txt"), "w")
    # for idx in range(len(all_fingerprints)):
    #     fingerprint = all_fingerprints[idx]
    #     fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
    #     _, filename = os.path.split(dataset.filenames[idx])
    #     filename = filename.split('.')[0] + ".png"
    #     f.write(f"{filename} {fingerprint_str}\n")
    # f.close()


def main():

    # operate on cifar10 dataset
    args.decoder_path = "./_output/cifar10/checkpoints/*pth"
    args.image_resolution = 32
    
    # ------------------------------------- #
    # identical
    args.identical_fingerprints = True
    
    # # non-identical
    # args.identical_fingerprints = False
    # ------------------------------------- #
    
    # generated_data_dir = "../edm/_samples/cifar10_embedded/" 
    generated_data_dir = "../edm/_samples/imagenet-64x64_non_cond_adm_4_bit" 
    note_outdir        = "../edm/datasets/embedded/imagenet_4_bit_non_id_encoder/note/"
    
    # process cifar10 dataset
    for i in tqdm(range(50)):
        args.data_dir    = os.path.join(generated_data_dir, f"{str(i).zfill(3)}" + "000",)
        # args.data_dir    = os.path.join(generated_data_dir, f"{str(i).zfill(5)}",)
        args.output_dir  = os.path.join(note_outdir, f"{str(i).zfill(5)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        load_decoder()
        load_data()
        extract_fingerprints()


if __name__ == "__main__":
    main()
