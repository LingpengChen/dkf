import torch

class UWBDecoder:
    def __init__(self):
        """
        Initialize UWB decoder
        No need for initialization parameters as M2 contains all necessary information
        """
        pass
        
    def decode(self, pose_samples, M2):
        """
        Decode UWB measurements using pose samples
        Args:
            pose_samples: sampled robot poses [batch_size, 3] (x, y, theta)
            M2: measurement matrix [batch_size, num_measurements, 5]
                each row: [tag_pos_robot_x, tag_pos_robot_y, anchor_pos_world_x, anchor_pos_world_y, d_measurement]
        Returns:
            reconstructed_distances: reconstructed UWB measurements [batch_size, num_measurements]
        """
        # batch_size = pose_samples.size(0)
        
        # Extract robot poses
        t_x = pose_samples[:, 0]  # [batch_size]
        t_y = pose_samples[:, 1]  # [batch_size]
        theta = pose_samples[:, 2] * (2 * torch.pi)  # Convert to radians [batch_size]
        
        # Compute rotation matrices for all samples in batch
        cos_theta = torch.cos(theta)  # [batch_size]
        sin_theta = torch.sin(theta)  # [batch_size]
        
        # Create rotation matrices [batch_size, 2, 2]
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=1),
            torch.stack([sin_theta, cos_theta], dim=1)
        ], dim=1)
        
        # Get tag positions in robot frame from M2
        tags_robot = M2[:, :, :2]  # [batch_size, num_measurements, 2]
        
        # Transform tag positions from robot frame to world frame
        # First rotate
        tags_world = torch.bmm(tags_robot, R.transpose(1, 2))  # [batch_size, num_measurements, 2]
        
        # Then translate
        translation = torch.stack([t_x, t_y], dim=1).unsqueeze(1)  # [batch_size, 1, 2]
        tags_world = tags_world + translation  # [batch_size, num_measurements, 2]
        
        # Get anchor positions from M2
        anchor_positions = M2[:, :, 2:4]  # [batch_size, num_measurements, 2]
        
        # Compute distances between anchors and transformed tags
        differences = anchor_positions - tags_world  # [batch_size, num_measurements, 2]
        reconstructed_distances = torch.norm(differences, dim=2)  # [batch_size, num_measurements]
        
        return reconstructed_distances

    def compute_reconstruction_loss(self, reconstructed_distances, M2):
        """
        Compute loss between reconstructed and actual measurements
        Args:
            reconstructed_distances: reconstructed measurements [batch_size, num_measurements]
            M2: measurement matrix containing actual measurements
        Returns:
            loss: abs between reconstructed and actual measurements
        """
        actual_distances = M2[:, :, 4]  # Get actual measurements
        # loss = torch.mean((reconstructed_distances - actual_distances) ** 2)
        loss = torch.mean(torch.abs((reconstructed_distances - actual_distances) ** 2))
        return loss