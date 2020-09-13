#!/usr/bin/python
#coding:utf-8
import ctypes


lib =  ctypes.cdll.LoadLibrary("darknet2.dll")

timer = lib.what_time_is_it_now
load_net = lib.load_network
load_meta = lib.get_metadata


def test_timer():
    time0 = timer()
    #print time0
    i = 10000*10000
    while i>0:
        i = i-1
    period = timer() - time0
    print "delay time %d sec" %(period)
    
if __name__ == "__main__":
    test_timer()
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    
    #net = load_net("cfg/tiny-yolo.cfg", "weights/tiny-yolo.weights", 0)
    #meta = load_meta("cfg/coco.data")
    
    raw_input("press enter to quit")
