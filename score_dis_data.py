import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils import CustomSubset
from diffusers import DDPMPipeline

########################################################################################################################
# Load Data
########################################################################################################################

def get_ddpm(path):
    """
    Load a pretrained Denoising Diffusion Probabilistic Model (DDPM) from the specified path.

    Args:
        path (str): The file path to the pretrained DDPM model.

    Returns:
        DDPMPipeline: The loaded DDPM pipeline.
    """
    print("loading pretrained ddpm from %s" %
          path)  # Print a message indicating the loading path.
    # Load the pretrained model using 16-bit precision.
    pipeline = DDPMPipeline.from_pretrained(path, torch_dtype=torch.float16)
    return pipeline  # Return the loaded DDPM pipeline.


def get_score_distillation_loss(pipe, image, steps=500):
    """
    Compute the score distillation loss for a given image using the provided DDPM pipeline.

    Args:
        pipe (DDPMPipeline): The DDPM pipeline.
        image (torch.Tensor): The input image tensor.
        steps (int): The number of steps for computing the loss.

    Returns:
        float: The computed score distillation loss.
    """
    # Initialize the loss to zero.
    all_loss = torch.zeros(image.shape[0])
    for i in range(image.shape[0]):  # Iterate over each image in the batch.
        # Get the noise scheduler from the DDPM pipeline.
        
        loss = 0.0
        
        noise_scheduler = pipe.scheduler

        # Replicate the current image `steps` times.

        # Each image is replicated steps times to create a batch of replicated images.
        # This helps in creating multiple noisy versions of the same image for different timesteps.
        replicated_image = image[i: i + 1].repeat(steps, 1, 1, 1)

        # Generate random noise with the same shape as the replicated image.
        noise = torch.randn(replicated_image.shape).to(image.device)

        timesteps = torch.randint(
            low=0, high=noise_scheduler.config.num_train_timesteps, size=(steps,)
        ).to(image.device)  # Generate random timesteps within the valid range.

        noisy_images = noise_scheduler.add_noise(
            replicated_image, noise, timesteps).half()  # Add noise to the replicated images at the
        # generated timesteps, converting to half precision.

        # Pass the noisy images and timesteps through the UNet model.
        model_out_dict = pipe.unet(noisy_images, timesteps)

        # Extract the model output from the dictionary.
        model_output = model_out_dict.sample

        # Compute the loss as the mean squared error between the noise and model output, summed
        # over the spatial dimensions and mean over the batch.

        # The loss is calculated as the mean squared error between the actual noise and the noise predicted by the UNet model.
        loss += (noise - model_output).square().sum(dim=(1, 2, 3)).mean()
        all_loss[i] = loss
    return all_loss  # Return the total computed loss.

def load_data(args):
    """
    Load data for training and testing.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    train_loader, test_loader = load_dataset(args)
    return train_loader, test_loader

def load_dataset(args):
    """
    Load dataset based on the specified dataset in args.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    if args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = load_cifar100(args)
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    return train_loader, test_loader
def score_dis_data(args, trainset, pipe):

    all_sample_losses = []
    trainset_permutation_inds = np.arange(len(trainset.targets))
    print("inds_size():",trainset_permutation_inds)
    batch_size = args.batch_size
    for batch_idx, batch_start_ind in enumerate(
            range(0, len(trainset.targets), batch_size)):
        print("batch_start_ind",batch_start_ind)
        # Get trainset indices for batch
        batch_inds = trainset_permutation_inds[batch_start_ind:
                                               batch_start_ind + batch_size]

        # Get batch inputs and targets, transform them appropriately
        transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(
            np.array(trainset.targets)[batch_inds].tolist())

        # Map to available device
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # Forward propagation, compute loss, get predictions
        with torch.no_grad():
            loss_sd = get_score_distillation_loss(pipe, inputs, steps=args.steps)
        # Update statistics and loss
        for j, index in enumerate(batch_inds):
            #sample_loss = loss[j].item() + sd_lbd * loss_sd[j].item()
            sample_loss = loss_sd[j].item()
            all_sample_losses.append((sample_loss, targets[j][1]))
    
    print("all_sample_losses:",all_sample_losses)
    all_sample_losses.sort(key=lambda x: x[0])  # Sort by loss
    top_half_indices = [index for _, index in all_sample_losses[:int(len(all_sample_losses) * args.rate)]]
    print("indices:",top_half_indices)
    new_trainset = CustomSubset(trainset, top_half_indices)


    return new_trainset

def load_cifar10(args):
    """
    Load CIFAR-10 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    pipe = get_ddpm(args.model_path)

    #pipe = nn.DataParallel(pipe)
    pipe = pipe.to(args.device)
    rate = args.rate

    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    new_trainset = score_dis_data(args, train_data, pipe)   
 
    
    train_loader = torch.utils.data.DataLoader(new_trainset, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader

def load_cifar100(args):
    """
    Load CIFAR-100 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader
