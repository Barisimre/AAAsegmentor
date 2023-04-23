import matplotlib.pyplot as plt
import numpy as np
import wandb


def visualize_sample_to_wandb(image, label, prediction, scores):

    image = image.cpu().numpy()
    label = label.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # Force a 2D example where at least one class is present
    while True:
        # Choose a random slice index
        slice_idx = np.random.randint(0, image.shape[2])

        # Extract the 2D slices
        image_slice = image[0, 0, slice_idx, :, :]
        label_slice = label[0, 0, slice_idx, :, :]
        pred_slice = prediction[0, 0, slice_idx, :, :]

        if len(np.unique(label_slice)) > 1:
            break

    # Plot the slices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Scores of this slice: {scores}')
    ax1.imshow(image_slice)
    ax1.set_title("Image")
    ax2.imshow(label_slice)
    ax2.set_title("Label")
    ax3.imshow(pred_slice)
    ax3.set_title("Prediction")

    # Remove axis ticks
    # for ax in [ax1, ax2, ax3]:
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    wandb.log({"plot": fig})
    plt.close(fig)
