height_restrictions = {
    "bottom": 34.4091666666942,
    "top": 35.850555555584236,
    "left": 32.09333333335901,
    "right": 34.677222222249966,
}

weather_restrictions = {
    "bottom": -4.4161896e-13,
    "top": 50.0,
    "left": -20.0,
    "right": 100.0,
}


def get_restrictions():
    return {
        "bottom": max(height_restrictions['bottom'], weather_restrictions['bottom']),
        "top": min(height_restrictions['top'], weather_restrictions['top']),
        "left": max(height_restrictions['left'], weather_restrictions['left']),
        "right": min(height_restrictions['right'], weather_restrictions['right']),
    }
