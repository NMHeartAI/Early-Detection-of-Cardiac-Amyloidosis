import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TreePlotConfig:
    CROP_SIZE: int = 1564  # square crop dimension
    LABEL_SIZE: int = 250  # magenta square behind letter label
    LABEL_COLOR: Tuple[int] = (122, 31, 108)
    LABEL_FONTCOLOR: Tuple[int] = (255.0, 255.0, 255.0)
    _LABEL_TO_CROP_DIST_V: int = (
        60  # v space between bottom right label and top left crop
    )
    _LABEL_TO_CROP_DIST_H: int = (
        60  # h space between bottom right label and top left crop
    )
    _HSPACE: int = 100  # space between adjacent crops
    _VSPACE_AFTER_CROP: int = 60  # v space between bottom of crop and start of next row

    # Top of each "row", 1 row per model
    row1: int = 0
    row2: int = (
        row1 + LABEL_SIZE + _LABEL_TO_CROP_DIST_V + CROP_SIZE + _VSPACE_AFTER_CROP
    )
    row3: int = (
        row2 + LABEL_SIZE + _LABEL_TO_CROP_DIST_V + CROP_SIZE + _VSPACE_AFTER_CROP
    )

    # Left side of each column, 1 column per disparity.
    # Only used by crop artists, not label artists
    col1: int = 0
    col2: int = col1 + CROP_SIZE + _HSPACE
    col3: int = col2 + CROP_SIZE + _HSPACE

    row_offset_list: Tuple[int] = (
        row1 + LABEL_SIZE + _LABEL_TO_CROP_DIST_V,
        row2 + LABEL_SIZE + _LABEL_TO_CROP_DIST_V,
        row3 + LABEL_SIZE + _LABEL_TO_CROP_DIST_V,
    )
    col_offset_list: Tuple[int] = (
        col1 + LABEL_SIZE + _LABEL_TO_CROP_DIST_H,
        col2 + LABEL_SIZE + _LABEL_TO_CROP_DIST_H,
        col3 + LABEL_SIZE + _LABEL_TO_CROP_DIST_H,
    )

    TITLE_NUDGE_UP: int = 50  # nudge so that titles sit just above crop
    TITLE_FONTCOLOR: Tuple[int] = (0.0, 0.0, 0.0)


def tree_plot(
    model_list: Tuple[str] = (
        "Mayo ATTR-CM Score",
        "EchoNet-LVH",
        "EchoGo Amyloidosis",
    ),
    aqp_names: Tuple[str] = (
        "_pprev_disparity",
        "_precision_disparity",
        "_fnr_disparity",
    ),
    metric_list: Tuple[str] = (
        "Demographic Parity",
        "Predictive Parity",
        "Equal Opportunity",
    ),
    src: Path = Path("./aequitas/"),
    dst: Path = Path("./aequitas/disparity_treemap.tiff"),
    config=TreePlotConfig,
) -> np.ndarray:
    image = np.ones((5802, 5512, 3), np.int8) * 255
    for model, row_offset in zip(model_list, config.row_offset_list):
        for metric, col_offset in zip(
            aqp_names,
            config.col_offset_list,
        ):
            raw_img = cv2.imread(str(src / f"{model}{metric}.tiff"))
            crop = raw_img[204 : 204 + config.CROP_SIZE, 3646 : 3646 + config.CROP_SIZE]

            image[
                row_offset : row_offset + config.CROP_SIZE,
                col_offset : col_offset + config.CROP_SIZE,
            ] = crop

    for top_px, bottom_px, letter in zip(
        [config.row1, config.row2, config.row3],
        [config.row2, config.row3, image.shape[0]],
        ["A", "B", "C"],
    ):
        # Place label background, then label, finally outline.
        cv2.rectangle(
            image,
            (0, top_px),
            (config.LABEL_SIZE, top_px + config.LABEL_SIZE),
            config.LABEL_COLOR,
            thickness=-3,
        )
        (width, height), baseline = cv2.getTextSize(
            letter, cv2.FONT_HERSHEY_DUPLEX, 7, thickness=14
        )
        cv2.putText(
            image,
            letter,
            (
                (config.LABEL_SIZE - width) // 2,
                (config.LABEL_SIZE - height) // 2 + height + top_px,
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            7,
            config.LABEL_FONTCOLOR,
            thickness=14,
        )
        cv2.rectangle(
            image, (0, top_px), (image.shape[1], bottom_px), (0, 0, 0), thickness=10
        )

    for col, title in enumerate(metric_list):
        (width, height), baseline = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_DUPLEX, 5, thickness=10
        )
        for row in range(3):
            cv2.putText(
                image,
                title,
                (
                    config.col_offset_list[col] + config.CROP_SIZE // 2 - width // 2,
                    config.row_offset_list[row] - config.TITLE_NUDGE_UP,
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                5,
                config.TITLE_FONTCOLOR,
                thickness=10,
            )

    plt.axis("off")
    _ = cv2.imwrite(
        str(dst), image, [cv2.IMWRITE_TIFF_XDPI, 300, cv2.IMWRITE_TIFF_YDPI, 300]
    )

    return image


if __name__ == "__main__":
    config = TreePlotConfig
    AEQDST = Path("./figures/aequitas")
    image = tree_plot(
        src=AEQDST, dst=AEQDST / "disparity_treemap.tiff", config=TreePlotConfig
    )
