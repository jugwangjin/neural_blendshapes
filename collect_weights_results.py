import os
import shutil

def collect_weights_results(given_directory):
    # Remove existing images in 'debug/loss_effects' directory
    loss_effects_dir = os.path.join('debug', 'loss_effects')
    if os.path.exists(loss_effects_dir):
        # Remove all files in the directory
        for filename in os.listdir(loss_effects_dir):
            file_path = os.path.join(loss_effects_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
    else:
        # Create the directory if it doesn't exist
        os.makedirs(loss_effects_dir)

    # List subdirectories that start with '1205'
    subdirectories = [
        d for d in os.listdir(given_directory)
        if d.startswith('1205') and os.path.isdir(os.path.join(given_directory, d))
    ]

    for subdirectory in subdirectories:
        for grid_value in [500, 1000, 1500]:
            # Construct the source paths for both blendshapes and ict_w_temp
            source_paths = {
                'blendshapes': os.path.join(
                    given_directory, subdirectory, 'stage_1', 'images', 'grid',
                    f'grid_{grid_value}_blendshapes_activation_0.png'
                ),
                'ict_w_temp': os.path.join(
                    given_directory, subdirectory, 'stage_1', 'images', 'grid',
                    f'grid_{grid_value}_ict_w_temp.png'
                )
            }

            # Construct the destination paths
            destination_paths = {
                'blendshapes': os.path.join('debug', 'loss_effects', 
                    f'grid_{grid_value}_{subdirectory}_bshapes_activation_0.png'),
                'ict_w_temp': os.path.join('debug', 'loss_effects',
                    f'grid_{grid_value}_{subdirectory}_ict_w_temp.png')
            }

            # Copy the images
            for img_type, source_path in source_paths.items():
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_paths[img_type])
                    print(f'Copied {source_path} to {destination_paths[img_type]}')
                else:
                    print(f'Source image not found: {source_path}')

# Example usage
given_directory = '/Bean/log/gwangjin/2024/neural_blendshapes'
collect_weights_results(given_directory)