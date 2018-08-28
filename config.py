'''
    Config file to define changable params
'''

data = {
    "location": "/media/matthewlee/DATA/data/Bushfire",
    "outputs": "/media/matthewlee/DATA/data/Bushfire/outputs",
    "fire": "with fire",
    "maybe_fire": "may have fire",
    "not_fire": "without fire"
}

data_abs = {
    "fire": "{}/{}".format(data['location'], data['fire']),
    "maybe_fire": "{}/{}".format(data['location'], data['maybe_fire']),
    "not_fire": "{}/{}".format(data['location'], data['not_fire'])
}
