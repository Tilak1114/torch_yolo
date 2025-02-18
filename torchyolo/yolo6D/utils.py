import torch
import torch.nn.functional as F

def rotation_encode(matrix):
    """
    Encode 3x3 rotation matrix to 6D representation.

    Args:
        matrix (torch.Tensor): (..., 3, 3) rotation matrix
    Returns:
        (torch.Tensor): (..., 6) 6D rotation representation
    """
    # Suppose matrix has shape (..., 3, 3), we separate the last 2 dims
    r1 = matrix[..., :, 0]  # First column
    r2 = matrix[..., :, 1]  # Second column

    # Normalize first vector
    r1_norm = F.normalize(r1, dim=-1)

    # Project second vector and normalize
    dot = torch.sum(r2 * r1_norm, dim=-1, keepdim=True)
    r2_proj = r2 - dot * r1_norm
    r2_norm = F.normalize(r2_proj, dim=-1)

    # Concatenate into 6D representation
    return torch.cat([r1_norm, r2_norm], dim=-1)


def rotation_decode(rot_6d):
    """
    Decode 6D rotation representation to 3x3 rotation matrix.

    Args:
        rot_6d (torch.Tensor): (..., 6) 6D rotation representation
    Returns:
        (torch.Tensor): (..., 3, 3) rotation matrix
    
    Raises:
        ValueError: If the resulting matrix is not orthonormal
    """
    # Split 6D representation
    r1 = rot_6d[..., :3]
    r2 = rot_6d[..., 3:]

    # Normalize first basis vector
    r1_norm = F.normalize(r1, dim=-1)

    # Orthogonalize second basis vector w.r.t the first
    dot = torch.sum(r2 * r1_norm, dim=-1, keepdim=True)
    r2_proj = r2 - dot * r1_norm
    r2_norm = F.normalize(r2_proj, dim=-1)

    # Third vector via cross product
    r3 = torch.cross(r1_norm, r2_norm, dim=-1)

    # Stack to form rotation matrix (..., 3, 3)
    matrix = torch.stack([r1_norm, r2_norm, r3], dim=-1)
    
    # Check orthonormality and print diagnostic info
    mTm = torch.matmul(matrix.transpose(-2, -1), matrix)
    error = (mTm - torch.eye(3, device=matrix.device)).abs().max()
    
    if error > 1e-3:  # Using a more lenient threshold during training
        print(f"Warning: High orthonormality error: {error:.6f}")
        print("Original 6D rotation:", rot_6d)
        print("Resulting matrix:", matrix)
        print("mTm:", mTm)
    
    return matrix

def translation_encode(trans, anchor_points, stride_tensor):
    """
    Encode absolute translation (cx, cy, tz) to relative offsets (δx, δy, δz).
    Args:
        trans: (B, N, 3) absolute translation (cx, cy, tz)
        anchor_points: (N, 2) anchor points
        stride_tensor: (N,) or (B, N) stride values
    Returns:
        (B, N, 3) encoded translation offsets
    """
    cx, cy, tz = trans.unbind(-1)
    
    # Handle both single stride and per-anchor stride cases
    if stride_tensor.dim() == 1:
        sx = sy = stride_tensor.unsqueeze(0)
    else:
        sx = sy = stride_tensor[..., 0]

    delta_x = (cx - anchor_points[..., 0]) / sx
    delta_y = (cy - anchor_points[..., 1]) / sy
    delta_z = torch.log(tz)

    return torch.stack((delta_x, delta_y, delta_z), dim=-1)

def translation_decode(pred_trans, anchor_points, stride_tensor):
    """
    Decode predicted translation offsets (δx, δy, δz) to absolute translation (cx, cy, tz).
    Args:
        pred_trans: (B, N, 3) predicted translation offsets
        anchor_points: (N, 2) anchor points
        stride_tensor: (N,) or (B, N) stride values
    Returns:
        (B, N, 3) decoded absolute translation
    """
    delta_x, delta_y, delta_z = pred_trans.unbind(-1)
    
    # Handle both single stride and per-anchor stride cases
    if stride_tensor.dim() == 1:
        sx = sy = stride_tensor.unsqueeze(0)
    else:
        sx = sy = stride_tensor[..., 0]

    cx = delta_x * sx + anchor_points[..., 0]
    cy = delta_y * sy + anchor_points[..., 1]
    tz = torch.exp(delta_z)

    return torch.stack((cx, cy, tz), dim=-1)

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes
        eps (float, optional): A small number to prevent division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: IoU matrix of shape (N, M) containing the IoU values for every pair of boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter = (torch.min(b1_x2[:, None], b2_x2) - torch.max(b1_x1[:, None], b2_x1)).clamp(0) * \
            (torch.min(b1_y2[:, None], b2_y2) - torch.max(b1_y1[:, None], b2_y1)).clamp(0)

    # Box areas
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # Union area
    union = area1[:, None] + area2 - inter

    # IoU
    iou = inter / (union + eps)
    return iou

def project_translation(cx_px, cy_px, tz, K_inv):
    """
    Project pixel coordinates and depth to 3D translation vector using inverse camera matrix.
    
    Args:
        cx_px (torch.Tensor): x-coordinate in pixels
        cy_px (torch.Tensor): y-coordinate in pixels 
        tz (torch.Tensor): z-depth
        K_inv (torch.Tensor): (3,3) inverse camera intrinsic matrix
    
    Returns:
        torch.Tensor: (3,) translation vector in camera coordinates
    """
    homogeneous = torch.tensor(
        [cx_px, cy_px, 1.0], 
        device=cx_px.device,
        dtype=torch.float32
    )  # shape (3,)
    K_inv_proj = K_inv @ homogeneous  # shape (3,)
    translation = tz * K_inv_proj  # shape (3,)
    return translation

def build_camera_matrix(fx, fy, cx, cy, device='cpu'):
    """
    Build camera intrinsic matrix K and its inverse.
    
    Args:
        fx (float): Focal length x
        fy (float): Focal length y
        cx (float): Principal point x
        cy (float): Principal point y
        device (str): Device to put tensors on
        
    Returns:
        tuple: (K, K_inv) camera matrix and its inverse
    """
    K = torch.tensor(
        [[fx, 0, cx], 
         [0, fy, cy], 
         [0, 0, 1]],
        dtype=torch.float32,
        device=device
    )
    K_inv = torch.linalg.inv(K)
    return K, K_inv
