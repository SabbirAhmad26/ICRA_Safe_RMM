from enum import Enum
import math


MOTO_NAMES = ["Harley-Davidson", "Kawasaki", "Yamaha"]
BICYCLE_NAMES = ["Gazelle", "Diamondback", "Bh"]
CAR_NAMES = ["Ford", "Bmw", "Toyota", "Nissan", "Mini", "Tesla", "Seat", "Lincoln", "Audi", "Carlamotors", "Citroen", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Jeep", "Nissan", "Dodge", "Mustang"]

'''
CAR_PROXIMITY_THRESH_NEAR_COLL = 4
CAR_PROXIMITY_THRESH_SUPER_NEAR = 7 # max number of feet between a car and another entity to build proximity relation
CAR_PROXIMITY_THRESH_VERY_NEAR = 10
CAR_PROXIMITY_THRESH_NEAR = 16
CAR_PROXIMITY_THRESH_VISIBLE = 25
MOTO_PROXIMITY_THRESH = 50
BICYCLE_PROXIMITY_THRESH = 50
PED_PROXIMITY_THRESH = 50
'''

#defines all types of actors which can exist
#order of enum values is important as this determines which function is called. DO NOT CHANGE ENUM ORDER
class ActorType(Enum):
    CAR = 0 
    ROAD = 1
    JUNCTION = 2
    
ACTOR_NAMES=['car', 'road', 'junction']

class Relations(Enum):
    isIn = 0
    near_coll = 1
    very_near = 2
    near = 3
    visible = 4
    connects = 5
    '''
    inDFrontOf = 6
    inSFrontOf = 7
    atDRearOf = 8
    atSRearOf = 9
    toLeftOf = 10
    toRightOf = 11
    '''

RELATION_COLORS = ["black", "red", "orange", "yellow", "green", "purple", "blue", 
                "sienna", "pink", "pink", "pink",  "turquoise", "turquoise", "turquoise", "violet", "violet"]
