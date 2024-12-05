import os
import shutil

def collect_weights_results(given_directory):
    # List subdirectories that start with '1205'
    subdirectories = [d for d in os.listdir(given_directory) if d.startswith('1205') and os.path.isdir(os.path.join(given_directory, d))]
    
    for subdirectory in subdirectories:
        for grid_value in [500, 1000, 1500]:
            # Construct the source image path
            source_image_path = os.path.join(given_directory, subdirectory, 'stage_1', 'images', 'grid', f'grid_{grid_value}_blendshapes_activation_0.png')
            
            # Construct the destination image path
            destination_image_path = os.path.join('debug', 'loss_effects', f'{subdirectory}_bshapes_activation_0_grid_{grid_value}.png')
            
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
            
            # Copy the image
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
                print(f'Copied {source_image_path} to {destination_image_path}')
            else:
                print(f'Source image not found: {source_image_path}')

# Example usage
given_directory = '/Bean/log/gwangjin/2024/neural_blendshapes'
collect_weights_results(given_directory)