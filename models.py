import pathlib


dice_types = {
    "d6": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six'],
        # keep index aligned with 'classes'
        "values":  [1,  2,  3,   4,   5,   6],
        "model_path": "d6_classifier.tflite",
        "dataset_dir": pathlib.Path("d6"),
    },
}
