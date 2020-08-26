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


spoutReceiver = None
spoutSender = None
spoutReceiverWidth = 640
spoutReceiverHeight = 480
textureReceiveID = 0
textureSendID = 0
receiving = False

def InitSpout(args):
    #Set global Variable
    global spoutReceiver
    global spoutReceiverWidth
    global spoutReceiverHeight
    global textureReceiveID
    global textureSendID
    global receiving
    global spoutSender

    # window details
    width = args.window_size[0]
    height = args.window_size[1]
    display = (width, height)

    # window setup
    pygame.init()
    pygame.display.set_caption('Spout Receiver')
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # OpenGL init
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)

    # init spout receiver
    spoutReceiverWidth = args.spout_size[0]
    spoutReceiverHeight = args.spout_size[1]
    # create spout receiver
    spoutReceiver = SpoutSDK.SpoutReceiver()
 
    if(args.spout_out):
        spoutSender = SpoutSDK.SpoutSender()
        spoutSender.CreateSender(args.spout_out, spoutReceiverWidth, spoutReceiverHeight, 0)

    # Its signature in c++ looks like this: bool pyCreateReceiver(const char* theName, unsigned int theWidth, unsigned int theHeight, bool bUseActive);
    spoutReceiver.pyCreateReceiver(args.spout_in, spoutReceiverWidth, spoutReceiverHeight, False)

    # create texture for spout receiver
    textureReceiveID = glGenTextures(1)
    textureSendID = glGenTextures(1)

    # initalise receiver texture
    glBindTexture(GL_TEXTURE_2D, textureReceiveID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # copy data into texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, spoutReceiverWidth, spoutReceiverHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

    #Ready to receive frame
    receiving = True
    # return spoutReceiver, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID


def GetSpoutFrame(spout_name):
    spout_receive = False
    if sys.version_info[1] == 5:
        spout_receive = spoutReceiver.pyReceiveTexture(spout_name, spoutReceiverWidth, spoutReceiverHeight,
                                                       textureReceiveID, GL_TEXTURE_2D, False, 0)
    else:
        spout_receive = spoutReceiver.pyReceiveTexture(spout_name, spoutReceiverWidth, spoutReceiverHeight,
                                                       textureReceiveID.item(), GL_TEXTURE_2D, False, 0)
    if spout_receive:
        # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, textureReceiveID)

        # copy pixel byte array from received texture
        spoutImage = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE,
                                   outputType=None)  # Using GL_RGB can use GL_RGBA
        glBindTexture(GL_TEXTURE_2D, 0)
        # swap width and height data around due to oddness with glGetTextImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
        spoutImage.shape = (spoutImage.shape[1], spoutImage.shape[0], spoutImage.shape[2])
        spoutImage = cv2.cvtColor(spoutImage, cv2.COLOR_BGR2RGB)

        # cv2.imshow(spout_name, spoutImage)

        return spoutImage


def SendSpoutFrame(frame, args):
    if not receiving:
        InitSpout(args)

    # frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channel = frame.shape

    # Copy the frame from the webcam into the sender texture
    glBindTexture(GL_TEXTURE_2D, textureSendID)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

    # Send texture to Spout
    # Its signature in C++ looks like this: bool SendTexture(GLuint TextureID, GLuint TextureTarget, unsigned int width, unsigned int height, bool bInvert=true, GLuint HostFBO = 0);

    if sys.version_info[1] == 5:
        spoutSender.SendTexture(textureSendID, GL_TEXTURE_2D, spoutReceiverWidth, spoutReceiverHeight, False, 0)
    else:
        spoutSender.SendTexture(textureSendID.item(), GL_TEXTURE_2D, spoutReceiverWidth, spoutReceiverHeight, False, 0)
