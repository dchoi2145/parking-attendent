#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.5  # meters
deck_attached_event = Event()

FT4 = 1.2192   # meters
FT8 = 2.4384   # meters

logging.basicConfig(level=logging.ERROR)

def param_deck_flow(name, value_str):
    value = int(value_str)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def move_up_on_detection(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        with Multiranger(scf) as mr:
            climbed = False
            while True:
                # read horizontal ranges (None means no measurement)
                fr = mr.front if mr.front is not None else 999
                ba = mr.back  if mr.back  is not None else 999
                le = mr.left  if mr.left  is not None else 999
                ri = mr.right if mr.right is not None else 999

                near = min(fr, ba, le, ri)

                if (near < FT4) and (not climbed):
                    # absolute move to 8 ft (Flow/Kalman absolute z)
                    mc.go_to(0, 0, FT8, velocity=0.5)
                    climbed = True

                time.sleep(0.1)

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        # Arm the Crazyflie
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        move_up_on_detection(scf)
