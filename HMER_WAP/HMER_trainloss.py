import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import matplotlib.pyplot as plt

# --- 1. Configuration ---
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 12

def plot_training_loss():
    epochs_evaluated = []
    training_losses = []

    print("Extracting training losses from checkpoints...")

    # --- 2. Extract Data ---
    for epoch in range(1, EPOCHS + 1):
        file_name = f'hmer_checkpoint_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(CHECKPOINT_DIR, file_name)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract the loss saved during training
            if 'loss' in checkpoint:
                epoch_loss = checkpoint['loss']
                epochs_evaluated.append(epoch)
                training_losses.append(epoch_loss)
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
            else:
                print(f"Warning: 'loss' key not found in {file_name}")
        else:
            print(f"Warning: Could not find {checkpoint_path}. Skipping.")

    # --- 3. Plot the Results ---
    if len(epochs_evaluated) > 0:
        print("\nGenerating Training Loss Plot...")
        
        # Recreate the exact TF ratios from the training loop
        teacher_forcing_ratios = {
            1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 
            6: 0.5, 7: 0.4, 8: 0.3, 9: 0.2, 10: 0.2, 
            11: 0.2, 12: 0.2
        }
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_evaluated, training_losses, marker='o', linestyle='-', color='purple', label='Training Loss')
        
        # Annotate each point with its Teacher Forcing Ratio
        for epoch, loss in zip(epochs_evaluated, training_losses):
            ratio = teacher_forcing_ratios.get(epoch, "N/A")
            ratio_str = f"{ratio:.1f}" if isinstance(ratio, float) else ratio
            
            plt.annotate(
                f'TF: {ratio_str}', 
                (epoch, loss), 
                textcoords="offset points", 
                xytext=(0, 15), 
                ha='center', 
                fontsize=9,
                color='indigo',
                fontweight='bold'
            )
        
        # Expand y-axis slightly so the top annotations don't clip past the edge
        y_min, y_max = min(training_losses), max(training_losses)
        plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.15)
        
        plt.title('Training Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Training Loss')
        
        # Ensure x-axis only shows whole numbers for the epochs we actually found
        plt.xticks(epochs_evaluated)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save and display
        plot_filename = "training_loss_plot.png"
        plt.savefig(plot_filename)
        print(f"Plot saved successfully as '{plot_filename}'")
        
        plt.show()
    else:
        print("No checkpoints were successfully loaded. Cannot generate plot.")

if __name__ == "__main__":
    plot_training_loss()