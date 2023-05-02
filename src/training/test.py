from src.constants import *
from src.utils.metrics import dice_scores
import numpy as np
import wandb
from src.utils.visualisations import visualize_sample_to_wandb


def test_single_epoch(model, test_loader):
    model.eval()
    scores = []

    largest_component = monai.transforms.KeepLargestConnectedComponent()

    visualised = False

    with torch.no_grad():
        for d in test_loader:
            img = d['img'].to(DEVICE)
            mask = d['mask'].to(DEVICE)

            out = monai.inferers.sliding_window_inference(img,
                                                          roi_size=CROP_SIZE,
                                                          sw_batch_size=BATCH_SIZE,
                                                          predictor=model,
                                                          overlap=0.5,
                                                          sw_device=DEVICE,
                                                          device="cpu",
                                                          progress=False,
                                                          mode="gaussian")
            out = torch.argmax(out, 1, keepdim=True).to(DEVICE)
            # out = largest_component(out).to(DEVICE)
            s = dice_scores(out, mask)
            scores.append(s)

            # Send one sample to wandb
            if not visualised:
                visualize_sample_to_wandb(img, mask, out, s)
                visualised = True

        scores = np.array(scores)
        scores = np.nan_to_num(scores, copy=True, nan=1.0)
        scores = np.sum(scores, axis=0) / scores.shape[0]

    test_score = np.sum(scores) / (len(scores) * 1.0)

    wandb.log({"test_score": test_score,
               "background": scores[0],
               "lumen": scores[1],
               "thrombus": scores[2]})

    model.train()

    return test_score
