# -*- coding: utf-8 -*-
from enum import Enum

class Activity(Enum):
    """For easy read and write specific activity lable"""
    SITTING = 1
    LYING = 2
    STANDING = 3
    WASHING_DISHES = 4
    VACUUMING = 5
    SWEEPING = 6
    WALKING = 7
    ASCENDING_STAIRS = 8
    DESCENDING_STAIRS = 9
    TREADMILL_RUNNING = 10
    BICYCLING_ON_50W = 11
    BICYCLING_ON_100W = 12
    ROPE_JUMPING = 13
    
        