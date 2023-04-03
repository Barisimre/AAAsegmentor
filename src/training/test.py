from src.constants import *
from src.utils.metrics import dice_scores
import numpy as np
import wandb


def test_single_epoch(model, test_loader):
    model.eval()
    scores = []

    largest_component = monai.transforms.KeepLargestConnectedComponent()

    with torch.no_grad():
        for d in test_loader:
            img = d['img'].to(DEVICE)
            mask = d['mask'].to(DEVICE)

            out = monai.inferers.sliding_window_inference(img,
                                                          roi_size=CROP_SIZE,
                                                          sw_batch_size=BATCH_SIZE,
                                                          predictor=model,
                                                          overlap=0.2,
                                                          sw_device=DEVICE,
                                                          device="cpu",
                                                          progress=False,
                                                          )
            out = torch.argmax(out, 1, keepdim=True)
            out = largest_component(out)

            scores.append(dice_scores(out, mask))

        scores = np.array(scores)
        scores = np.nan_to_num(scores, copy=True, nan=1.0)
        scores = np.nansum(scores, axis=0) / scores.shape[0]

    test_score = np.sum(scores) / (len(scores) * 1.0)

    wandb.log({"test_score": test_score,
               "background": scores[0],
               "lumen": scores[1],
               "thrombus": scores[2]})

    return test_score
