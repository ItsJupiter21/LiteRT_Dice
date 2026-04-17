import pathlib


DICE_TYPES = {
    "d6": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six'],
        "values":  [1,     2,     3,       4,      5,      6],
        # keep index aligned with 'classes'
        "model_path": "d6_classifier.tflite",
        "model_path_keras": "d6_classifier.keras",
        "dataset_dir": pathlib.Path("d6"),
    },
    "d6_pips": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six'],
        "values":  [1,     2,     3,       4,      5,      6],
        # keep index aligned with 'classes'
        "model_path": "d6_pips_classifier.tflite",
        "model_path_keras": "d6_pips_classifier.keras",
        "dataset_dir": pathlib.Path("d6_pips"),
    },

    "d8": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'],
        "values":  [1,     2,     3,       4,      5,      6,     7,       8],
        # keep index aligned with 'classes'
        "model_path": "d8_classifier.tflite",
        "model_path_keras": "d8_classifier.keras",
        "dataset_dir": pathlib.Path("d8"),
    },
    "d10": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero'],
        "values":  [1,     2,     3,       4,      5,      6,     7,       8,       9,      0],
        # keep index aligned with 'classes'
        "model_path": "d10_classifier.tflite",
        "model_path_keras": "d10_classifier.keras",

        "dataset_dir": pathlib.Path("d10"),
    },

    "d12": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'],
        "values":  [1,     2,     3,       4,      5,      6,     7,       8,       9,      10,    11,       12],
        # keep index aligned with 'classes'
        "model_path": "d12_classifier.tflite",
        "model_path_keras": "d12_classifier.keras",

        "dataset_dir": pathlib.Path("d12"),
    },
    "d20": {
        "classes": ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'],
        "values":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        # keep index aligned with 'classes'
        "model_path": "d20_classifier.tflite",
        "model_path_keras": "d20_classifier.keras",
        "dataset_dir": pathlib.Path("d20"),
    },

}
