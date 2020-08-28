# Add relative directory ../Library to import path, so we can import the SpoutSDK.pyd library. Feel free to remove these if you put the SpoutSDK.pyd file in the same directory as the python scripts.
import sys
sys.path.append('../Library')

import argparse
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2

#TODO : With pygame you can only instantiate once, beacuse pygame only supports one window per process. Replace with something like pyglet

class Spout:
    def __init__(self,args):
        #Set global Variable
        self.spoutReceiver = None
        self.spoutReceiverWidth = 640
        self.spoutReceiverHeight = 480
        self.textureReceiveID = 0
        self.textureSendID =0
        self.receiving = False
        self.spoutSenders = []

        # window details
        self.width = args.window_size[0]
        self.height = args.window_size[1]
        self.display = (self.width, self.height)

        #window setup
        pygame.init()
        pygame.display.set_caption('Spout Receiver')
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
       

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,self.width,self.height,0,1,-1)
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.0,0.0,0.0,0.0)
        glEnable(GL_TEXTURE_2D)

        # init spout receiver
        self.spoutReceiverWidth = args.spout_size[0]
        self.spoutReceiverHeight = args.spout_size[1]
        # create spout receiver
        self.spoutReceiver = SpoutSDK.SpoutReceiver()
        # Its signature in c++ looks like this: bool pyCreateReceiver(const char* theName, unsigned int theWidth, unsigned int theHeight, bool bUseActive);
        self.spoutReceiver.pyCreateReceiver(args.spout_in, self.spoutReceiverWidth, self.spoutReceiverHeight, False)
        if args.spout_out:
            for x in args.spout_out:
                sender = SpoutSDK.SpoutSender()
                sender.CreateSender(x, self.spoutReceiverWidth, self.spoutReceiverHeight, 0)
                self.spoutSenders.append(sender)



        # create texture for spout receiver
        self.textureReceiveID = glGenTextures(1)
        self.textureSendID = glGenTextures(1)

        # initalise receiver texture
        glBindTexture(GL_TEXTURE_2D, self.textureReceiveID)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # copy data into texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.spoutReceiverWidth, self.spoutReceiverHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        #Ready to receive frame
        self.receiving = True
        # return spoutReceiver, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID


    def GetSpoutFrame(self,spout_name):
        spout_receive = False
        if sys.version_info[1] == 5:
            spout_receive = self.spoutReceiver.pyReceiveTexture(spout_name, self.spoutReceiverWidth, self.spoutReceiverHeight,
                                                        self.textureReceiveID, GL_TEXTURE_2D, False, 0)
        else:
            spout_receive = self.spoutReceiver.pyReceiveTexture(spout_name, self.spoutReceiverWidth, self.spoutReceiverHeight,
                                                        self.textureReceiveID.item(), GL_TEXTURE_2D, False, 0)
        if spout_receive:
            # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glBindTexture(GL_TEXTURE_2D, self.textureReceiveID)

            # copy pixel byte array from received texture
            spoutImage = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE,
                                    outputType=None)  # Using GL_RGB can use GL_RGBA
            glBindTexture(GL_TEXTURE_2D, 0)
            # swap width and height data around due to oddness with glGetTextImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
            spoutImage.shape = (spoutImage.shape[1], spoutImage.shape[0], spoutImage.shape[2])
            spoutImage = cv2.cvtColor(spoutImage, cv2.COLOR_BGR2RGB)

            # cv2.imshow(spout_name, spoutImage)

            return spoutImage


    def SendSpoutFrame(self,frames, args):
        if not self.receiving:
            InitSpout(args)
        if not isinstance(frames, list):
            frames = [frames]
        for i,frame in enumerate(frames):
            # frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow("sppout",frame)
            height, width, channel = frame.shape

            # Copy the frame from the webcam into the sender texture
            glBindTexture(GL_TEXTURE_2D, self.textureSendID)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

            # Send texture to Spout
            # Its signature in C++ looks like this: bool SendTexture(GLuint TextureID, GLuint TextureTarget, unsigned int width, unsigned int height, bool bInvert=true, GLuint HostFBO = 0);
            if sys.version_info[1] == 5:
                self.spoutSenders[i].SendTexture(self.textureSendID, GL_TEXTURE_2D, self.spoutReceiverWidth, self.spoutReceiverHeight, False, 0)
            else:
                self.spoutSenders[i].SendTexture(self.textureSendID.item(), GL_TEXTURE_2D, self.spoutReceiverWidth, self.spoutReceiverHeight, False, 0)
