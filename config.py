'''
    Config file to define changable params
'''

data = {
    "location": "D:/data/GOES",
    "outputs": "D:/data/GOES/outputs",
    "fire": "with_fire",
    "maybe_fire": "maybe_fire",
    "not_fire": "without_fire"
}

data_abs = {
    "fire": "{}/{}".format(data['location'], data['fire']),
    "maybe_fire": "{}/{}".format(data['location'], data['maybe_fire']),
    "not_fire": "{}/{}".format(data['location'], data['not_fire'])
}
