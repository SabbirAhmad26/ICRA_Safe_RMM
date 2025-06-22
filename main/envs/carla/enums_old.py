from enum import Enum

class Behavior(Enum):
    """
    todo
    """
    EMERGENCY_STOP = -1
    KEEP_LANE = 0
    CHANGE_LEFT = 1
    CHANGE_RIGHT = 2
    BRK_0 = 3 # [-0.5, 0]
    ACC_1 = 4
    BRK_1 = 5
    ACC_2 = 6
    BRK_2 = 7
    ACC_3 = 8
    BRK_3 = 9
    ACC_4 = 10
    BRK_4 = 11
    ACC_5 = 12
    BRK_5 = 13

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 


class V_Type(Enum):
    CAV = 0
    HAZV = 1
    NCAV = 2
    

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving
    from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6