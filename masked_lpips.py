import torch
import lpips
from PIL import Image
import torchvision.transforms.functional as TF

def masked_lpips(img1, img2, mask, net='vgg'):
    """
    Calculate LPIPS score for masked regions of images.
    
    Args:
        img1 (torch.Tensor): First image tensor of shape (B, C, H, W)
        img2 (torch.Tensor): Second image tensor of shape (B, C, H, W)
        mask (torch.Tensor): Binary mask tensor of shape (B, 1, H, W) or (B, C, H, W)
        net (str): Network to use for LPIPS calculation ('vgg' or 'alex')
    
    Returns:
        torch.Tensor: Masked LPIPS score
    """
    # Ensure inputs are on the same device
    device = img1.device
    
    # Initialize LPIPS model with spatial output
    lpips_model = lpips.LPIPS(net=net, spatial=True).to(device)
    
    # Ensure mask has the right shape (if single channel, expand to match image channels)
    if mask.shape[1] == 1:
        mask = mask.repeat(1, 3, 1, 1)
    
    with torch.no_grad():
        # Calculate spatial LPIPS map
        lpips_map = lpips_model(img1, img2)
        
        # Apply mask and calculate mean over masked region
        masked_score = torch.sum(lpips_map * mask) / torch.sum(mask)
        
    return masked_score

def load_and_preprocess_image(image_path):
    """Load and preprocess image for LPIPS calculation."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0)
    return img_tensor

def load_and_preprocess_mask(mask_path):
    """Load and preprocess mask."""
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask_tensor = TF.to_tensor(mask).unsqueeze(0)
    # Binarize the mask
    mask_tensor[mask_tensor > 0.5] = 1
    mask_tensor[mask_tensor <= 0.5] = 0
    return mask_tensor

if __name__ == "__main__":
    # Path definitions
    mask_path = '/project/hentci/mip-nerf-360/trigger_kitchen_fox/DSCF0656_mask.JPG'
    # image1_path = '/project/hentci/GS-backdoor/models/kitchen_-0.15/log_images/iteration_030000.png'
    image1_path = '/project/hentci/GS-backdoor/IPA-test/eval_kitchen_step2/log_images/iteration_030000.png'
    image2_path = '/project/hentci/mip-nerf-360/trigger_kitchen_fox/DSCF0656.JPG'
    
    # Load and preprocess images and mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img1_tensor = load_and_preprocess_image(image1_path).to(device)
    img2_tensor = load_and_preprocess_image(image2_path).to(device)
    mask_tensor = load_and_preprocess_mask(mask_path).to(device)
    
    # Calculate masked LPIPS
    score = masked_lpips(img1_tensor, img2_tensor, mask_tensor)
    print(f"Masked LPIPS score: {score.item():.4f}")
    
    # Also calculate unmasked LPIPS for comparison
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    with torch.no_grad():
        unmasked_score = lpips_model(img1_tensor, img2_tensor)
    print(f"Unmasked LPIPS score: {unmasked_score.item():.4f}")