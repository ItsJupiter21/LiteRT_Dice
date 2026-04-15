

import pathlib


dice_types = {
    "d6": {
        "classes": ['five', 'four', 'one', 'six', 'three', 'two'],
        "dataset_dir": pathlib.Path("d6"),
        "model_path": "d6_classifier.tflite"
    },
}
