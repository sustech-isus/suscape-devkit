# coding:utf-8

'''
Name  : obj_color.py
Author: Tim Liu

'''


classname_to_color = {
        "Car": '#86af49',
        "Pedestrian": '#ff0000',
        "Van": '#00ff00',
        "Bus": '#ffff00',
        "Truck": '#00ffff',
        "ScooterRider": '#ff8800',
        "Scooter": '#aaaa00',
        "BicycleRider": '#88ff00',
        "Bicycle": '#ff8800',
        "Motorcycle": '#aaaa00',
        "MotorcyleRider":'#ff8800',
        "PoliceCar": '#86af49',
        "TourCar": '#86af49',
        "RoadWorker": '#ff0000',
        "Child" : '#ff0000',
        "Cone": '#ff0000',
        "FireHydrant": '#ff0000',
        "SaftyTriangle": '#ff0000',
        "PlatformCart": '#ff0000',
        "ConstructionCart": '#ff0000',
        "RoadBarrel": '#ff0000',
        "TrafficBarrier": '#ff0000',
        "LongVehicle": '#ff0000',
        "BicycleGroup": '#ff0000',
        "ConcreteTruck": '#00ffff',
        "Tram": '#00ffff',
        "Excavator": '#00ffff',
        "Animal": '#00aaff',
        "TrashCan": '#00aaff',
        "ForkLift": '#00aaff',
        "Trimotorcycle": '#00aaff',
        "FreightTricycle": '#00aaff',
        "Crane": '#00aaff',
        "RoadRoller": '#00aaff',
        "Bulldozer":  '#00aaff',
        "DontCare": '#00ff88',
        "Misc": '#008888',
}

def rgb_to_hex(rgb):
    '''
    transfer the rgb value to hex value
    :param rgb: cell,the RGB value of the color
    :return: color in hex
    '''
    color = '#'
    for i in rgb:
        num = i
        color += str(hex(num))[-2:].replace('x','0').upper()
    return color

def hex_to_rgb(hex):
    '''
    transfer the hex value to rgb value
    :param hex: the color in hex format
    :return: color in rgb
    '''
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7],16)
    rgb = (r,g,b)
    # rgb = (b, g, r)
    return rgb

if __name__ == '__main__':
    # rgb = (255,255,0)
    # color = rgb_to_hex(rgb)
    hex_color = "#FFFF00"
    color = hex_to_rgb(hex_color)
    print("aaaa")