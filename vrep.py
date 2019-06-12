from __future__ import division
from __future__ import absolute_import
import platform
import struct
import sys
import os
import ctypes as ct
from vrepConst import *

#load library
libsimx = None
try:
    file_extension = u'.so'
    if platform.system() ==u'cli':
        file_extension = u'.dll'
    elif platform.system() ==u'Windows':
        file_extension = u'.dll'
    elif platform.system() == u'Darwin':
        file_extension = u'.dylib'
    else:
        file_extension = u'.so'
    libfullpath = os.path.join(os.path.dirname(__file__), u'remoteApi' + file_extension)
    libsimx = ct.CDLL(libfullpath)
except:
    print u'----------------------------------------------------'
    print u'The remoteApi library could not be loaded. Make sure'
    print u'it is located in the same folder as "vrep.py", or'
    print u'appropriately adjust the file "vrep.py"'
    print u'----------------------------------------------------'
    print u''

#ctypes wrapper prototypes
c_GetJointPosition          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetJointPosition", libsimx))
c_SetJointPosition          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32)((u"simxSetJointPosition", libsimx))
c_GetJointMatrix            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetJointMatrix", libsimx))
c_SetSphericalJointMatrix   = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxSetSphericalJointMatrix", libsimx))
c_SetJointTargetVelocity    = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32)((u"simxSetJointTargetVelocity", libsimx))
c_SetJointTargetPosition    = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32)((u"simxSetJointTargetPosition", libsimx))
c_GetJointForce             = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetJointForce", libsimx))
c_SetJointForce             = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32)((u"simxSetJointForce", libsimx))
c_ReadForceSensor           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int32)((u"simxReadForceSensor", libsimx))
c_BreakForceSensor          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32)((u"simxBreakForceSensor", libsimx))
c_ReadVisionSensor          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.POINTER(ct.c_float)), ct.POINTER(ct.POINTER(ct.c_int32)), ct.c_int32)((u"simxReadVisionSensor", libsimx))
c_GetObjectHandle           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetObjectHandle", libsimx))
c_GetVisionSensorImage      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_byte)), ct.c_ubyte, ct.c_int32)((u"simxGetVisionSensorImage", libsimx))
c_SetVisionSensorImage      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_byte), ct.c_int32, ct.c_ubyte, ct.c_int32)((u"simxSetVisionSensorImage", libsimx))
c_GetVisionSensorDepthBuffer= ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32)((u"simxGetVisionSensorDepthBuffer", libsimx))
c_GetObjectChild            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetObjectChild", libsimx))
c_GetObjectParent           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetObjectParent", libsimx))
c_ReadProximitySensor       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_float), ct.c_int32)((u"simxReadProximitySensor", libsimx))
c_LoadModel                 = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_ubyte, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxLoadModel", libsimx))
c_LoadUI                    = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_ubyte, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_int32)), ct.c_int32)((u"simxLoadUI", libsimx))
c_LoadScene                 =  ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_ubyte, ct.c_int32)((u"simxLoadScene", libsimx))
c_StartSimulation           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32)((u"simxStartSimulation", libsimx))
c_PauseSimulation           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32)((u"simxPauseSimulation", libsimx))
c_StopSimulation            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32)((u"simxStopSimulation", libsimx))
c_GetUIHandle               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetUIHandle", libsimx))
c_GetUISlider               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetUISlider", libsimx))
c_SetUISlider               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32)((u"simxSetUISlider", libsimx))
c_GetUIEventButton          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetUIEventButton", libsimx))
c_GetUIButtonProperty       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetUIButtonProperty", libsimx))
c_SetUIButtonProperty       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32)((u"simxSetUIButtonProperty", libsimx))
c_AddStatusbarMessage       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32)((u"simxAddStatusbarMessage", libsimx))
c_AuxiliaryConsoleOpen      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxAuxiliaryConsoleOpen", libsimx))
c_AuxiliaryConsoleClose     = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32)((u"simxAuxiliaryConsoleClose", libsimx))
c_AuxiliaryConsolePrint     = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32)((u"simxAuxiliaryConsolePrint", libsimx))
c_AuxiliaryConsoleShow      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_ubyte, ct.c_int32)((u"simxAuxiliaryConsoleShow", libsimx))
c_GetObjectOrientation      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetObjectOrientation", libsimx))
c_GetObjectQuaternion       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetObjectQuaternion", libsimx))
c_GetObjectPosition         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetObjectPosition", libsimx))
c_SetObjectOrientation      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxSetObjectOrientation", libsimx))
c_SetObjectQuaternion       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxSetObjectQuaternion", libsimx))
c_SetObjectPosition         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxSetObjectPosition", libsimx))
c_SetObjectParent           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_ubyte, ct.c_int32)((u"simxSetObjectParent", libsimx))
c_SetUIButtonLabel          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_char), ct.c_int32)((u"simxSetUIButtonLabel", libsimx))
c_GetLastErrors             = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_char)), ct.c_int32)((u"simxGetLastErrors", libsimx))
c_GetArrayParameter         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetArrayParameter", libsimx))
c_SetArrayParameter         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxSetArrayParameter", libsimx))
c_GetBooleanParameter       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_ubyte), ct.c_int32)((u"simxGetBooleanParameter", libsimx))
c_SetBooleanParameter       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_ubyte, ct.c_int32)((u"simxSetBooleanParameter", libsimx))
c_GetIntegerParameter       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetIntegerParameter", libsimx))
c_SetIntegerParameter       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32)((u"simxSetIntegerParameter", libsimx))
c_GetFloatingParameter      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetFloatingParameter", libsimx))
c_SetFloatingParameter      = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32)((u"simxSetFloatingParameter", libsimx))
c_GetStringParameter        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.POINTER(ct.c_char)), ct.c_int32)((u"simxGetStringParameter", libsimx))
c_GetCollisionHandle        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetCollisionHandle", libsimx))
c_GetDistanceHandle         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetDistanceHandle", libsimx))
c_GetCollectionHandle       = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetCollectionHandle", libsimx))
c_ReadCollision             = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_ubyte), ct.c_int32)((u"simxReadCollision", libsimx))
c_ReadDistance              = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxReadDistance", libsimx))
c_RemoveObject              = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32)((u"simxRemoveObject", libsimx))
c_RemoveModel               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32)((u"simxRemoveModel", libsimx))
c_RemoveUI                  = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32)((u"simxRemoveUI", libsimx))
c_CloseScene                = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32)((u"simxCloseScene", libsimx))
c_GetObjects                = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_int32)), ct.c_int32)((u"simxGetObjects", libsimx))
c_DisplayDialog             = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_char), ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxDisplayDialog", libsimx))
c_EndDialog                 = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32)((u"simxEndDialog", libsimx))
c_GetDialogInput            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.POINTER(ct.c_char)), ct.c_int32)((u"simxGetDialogInput", libsimx))
c_GetDialogResult           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetDialogResult", libsimx))
c_CopyPasteObjects          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32, ct.POINTER(ct.POINTER(ct.c_int32)), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxCopyPasteObjects", libsimx))
c_GetObjectSelection        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.POINTER(ct.c_int32)), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetObjectSelection", libsimx))
c_SetObjectSelection        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32, ct.c_int32)((u"simxSetObjectSelection", libsimx))
c_ClearFloatSignal          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32)((u"simxClearFloatSignal", libsimx))
c_ClearIntegerSignal        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32)((u"simxClearIntegerSignal", libsimx))
c_ClearStringSignal         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32)((u"simxClearStringSignal", libsimx))
c_GetFloatSignal            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetFloatSignal", libsimx))
c_GetIntegerSignal          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetIntegerSignal", libsimx))
c_GetStringSignal           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.POINTER(ct.c_ubyte)), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetStringSignal", libsimx))
c_SetFloatSignal            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_float, ct.c_int32)((u"simxSetFloatSignal", libsimx))
c_SetIntegerSignal          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32, ct.c_int32)((u"simxSetIntegerSignal", libsimx))
c_SetStringSignal           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_ubyte), ct.c_int32, ct.c_int32)((u"simxSetStringSignal", libsimx))
c_AppendStringSignal        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_ubyte), ct.c_int32, ct.c_int32)((u"simxAppendStringSignal", libsimx))
c_WriteStringStream         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_ubyte), ct.c_int32, ct.c_int32)((u"simxWriteStringStream", libsimx))
c_GetObjectFloatParameter   = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetObjectFloatParameter", libsimx))
c_SetObjectFloatParameter   = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32)((u"simxSetObjectFloatParameter", libsimx))
c_GetObjectIntParameter     = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetObjectIntParameter", libsimx))
c_SetObjectIntParameter     = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32)((u"simxSetObjectIntParameter", libsimx))
c_GetModelProperty          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetModelProperty", libsimx))
c_SetModelProperty          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32)((u"simxSetModelProperty", libsimx))
c_Start                     = ct.CFUNCTYPE(ct.c_int32,ct.POINTER(ct.c_char), ct.c_int32, ct.c_ubyte, ct.c_ubyte, ct.c_int32, ct.c_int32)((u"simxStart", libsimx))
c_Finish                    = ct.CFUNCTYPE(None, ct.c_int32)((u"simxFinish", libsimx))
c_GetPingTime               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_int32))((u"simxGetPingTime", libsimx))
c_GetLastCmdTime            = ct.CFUNCTYPE(ct.c_int32,ct.c_int32)((u"simxGetLastCmdTime", libsimx))
c_SynchronousTrigger        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32)((u"simxSynchronousTrigger", libsimx))
c_Synchronous               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_ubyte)((u"simxSynchronous", libsimx))
c_PauseCommunication        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_ubyte)((u"simxPauseCommunication", libsimx))
c_GetInMessageInfo          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32))((u"simxGetInMessageInfo", libsimx))
c_GetOutMessageInfo         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32))((u"simxGetOutMessageInfo", libsimx))
c_GetConnectionId           = ct.CFUNCTYPE(ct.c_int32,ct.c_int32)((u"simxGetConnectionId", libsimx))
c_CreateBuffer              = ct.CFUNCTYPE(ct.POINTER(ct.c_ubyte), ct.c_int32)((u"simxCreateBuffer", libsimx))
c_ReleaseBuffer             = ct.CFUNCTYPE(None, ct.c_void_p)((u"simxReleaseBuffer", libsimx))
c_TransferFile              = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_char), ct.c_int32, ct.c_int32)((u"simxTransferFile", libsimx))
c_EraseFile                 = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.c_int32)((u"simxEraseFile", libsimx))
c_GetAndClearStringSignal   = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.POINTER(ct.c_ubyte)), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxGetAndClearStringSignal", libsimx))
c_ReadStringStream          = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.POINTER(ct.c_ubyte)), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxReadStringStream", libsimx))
c_CreateDummy               = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_float, ct.POINTER(ct.c_ubyte), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxCreateDummy", libsimx))
c_Query                     = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.c_ubyte), ct.c_int32, ct.POINTER(ct.c_char), ct.POINTER(ct.POINTER(ct.c_ubyte)), ct.POINTER(ct.c_int32), ct.c_int32)((u"simxQuery", libsimx))
c_GetObjectGroupData        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_int32)), ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_int32)), ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_float)), ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_char)), ct.c_int32)((u"simxGetObjectGroupData", libsimx))
c_GetObjectVelocity         = ct.CFUNCTYPE(ct.c_int32,ct.c_int32, ct.c_int32, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.c_int32)((u"simxGetObjectVelocity", libsimx))
c_CallScriptFunction        = ct.CFUNCTYPE(ct.c_int32,ct.c_int32,ct.POINTER(ct.c_char),ct.c_int32,ct.POINTER(ct.c_char),ct.c_int32,ct.POINTER(ct.c_int32),ct.c_int32,ct.POINTER(ct.c_float),ct.c_int32,ct.POINTER(ct.c_char),ct.c_int32,ct.POINTER(ct.c_ubyte),ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_int32)),ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_float)),ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_char)),ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_ubyte)),ct.c_int32)((u"simxCallScriptFunction", libsimx))

#API functions
def simxGetJointPosition(clientID, jointHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    position = ct.c_float()
    return c_GetJointPosition(clientID, jointHandle, ct.byref(position), operationMode), position.value

def simxSetJointPosition(clientID, jointHandle, position, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetJointPosition(clientID, jointHandle, position, operationMode)

def simxGetJointMatrix(clientID, jointHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    matrix = (ct.c_float*12)()
    ret = c_GetJointMatrix(clientID, jointHandle, matrix, operationMode)
    arr = []
    for i in xrange(12):
        arr.append(matrix[i])
    return ret, arr

def simxSetSphericalJointMatrix(clientID, jointHandle, matrix, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    matrix = (ct.c_float*12)(*matrix)
    return c_SetSphericalJointMatrix(clientID, jointHandle, matrix, operationMode)

def simxSetJointTargetVelocity(clientID, jointHandle, targetVelocity, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetJointTargetVelocity(clientID, jointHandle, targetVelocity, operationMode)

def simxSetJointTargetPosition(clientID, jointHandle, targetPosition, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetJointTargetPosition(clientID, jointHandle, targetPosition, operationMode)

def simxJointGetForce(clientID, jointHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    force = ct.c_float()
    return c_GetJointForce(clientID, jointHandle, ct.byref(force), operationMode), force.value

def simxGetJointForce(clientID, jointHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    force = ct.c_float()
    return c_GetJointForce(clientID, jointHandle, ct.byref(force), operationMode), force.value

def simxSetJointForce(clientID, jointHandle, force, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    return c_SetJointForce(clientID, jointHandle, force, operationMode)

def simxReadForceSensor(clientID, forceSensorHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    state = ct.c_ubyte()
    forceVector  = (ct.c_float*3)()
    torqueVector = (ct.c_float*3)()
    ret = c_ReadForceSensor(clientID, forceSensorHandle, ct.byref(state), forceVector, torqueVector, operationMode)
    arr1 = []
    for i in xrange(3):
        arr1.append(forceVector[i])
    arr2 = []
    for i in xrange(3):
        arr2.append(torqueVector[i])
    #if sys.version_info[0] == 3:
    #    state=state.value
    #else:
    #    state=ord(state.value)
    return ret, state.value, arr1, arr2

def simxBreakForceSensor(clientID, forceSensorHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    return c_BreakForceSensor(clientID, forceSensorHandle, operationMode)

def simxReadVisionSensor(clientID, sensorHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    detectionState = ct.c_ubyte()
    auxValues      = ct.POINTER(ct.c_float)()
    auxValuesCount = ct.POINTER(ct.c_int)()
    ret = c_ReadVisionSensor(clientID, sensorHandle, ct.byref(detectionState), ct.byref(auxValues), ct.byref(auxValuesCount), operationMode)

    auxValues2 = []
    if ret == 0:
        s = 0
        for i in xrange(auxValuesCount[0]):
            auxValues2.append(auxValues[s:s+auxValuesCount[i+1]])
            s += auxValuesCount[i+1]

        #free C buffers
        c_ReleaseBuffer(auxValues)
        c_ReleaseBuffer(auxValuesCount)

    return ret, bool(detectionState.value!=0), auxValues2

def simxGetObjectHandle(clientID, objectName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    handle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(objectName) is unicode):
        objectName=objectName.encode(u'utf-8')
    return c_GetObjectHandle(clientID, objectName, ct.byref(handle), operationMode), handle.value

def simxGetVisionSensorImage(clientID, sensorHandle, options, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    resolution = (ct.c_int*2)()
    c_image  = ct.POINTER(ct.c_byte)()
    bytesPerPixel = 3
    if (options and 1) != 0:
        bytesPerPixel = 1
    ret = c_GetVisionSensorImage(clientID, sensorHandle, resolution, ct.byref(c_image), options, operationMode)

    reso = []
    image = []
    if (ret == 0):
        image = [None]*resolution[0]*resolution[1]*bytesPerPixel
        for i in xrange(resolution[0] * resolution[1] * bytesPerPixel):
            image[i] = c_image[i]
        for i in xrange(2):
            reso.append(resolution[i])
    return ret, reso, image

def simxSetVisionSensorImage(clientID, sensorHandle, image, options, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    size = len(image)
    image_bytes  = (ct.c_byte*size)(*image)
    return c_SetVisionSensorImage(clientID, sensorHandle, image_bytes, size, options, operationMode)

def simxGetVisionSensorDepthBuffer(clientID, sensorHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    c_buffer  = ct.POINTER(ct.c_float)()
    resolution = (ct.c_int*2)()
    ret = c_GetVisionSensorDepthBuffer(clientID, sensorHandle, resolution, ct.byref(c_buffer), operationMode)
    reso = []
    buffer = []
    if (ret == 0):
        buffer = [None]*resolution[0]*resolution[1]
        for i in xrange(resolution[0] * resolution[1]):
            buffer[i] = c_buffer[i]
        for i in xrange(2):
            reso.append(resolution[i])
    return ret, reso, buffer

def simxGetObjectChild(clientID, parentObjectHandle, childIndex, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    childObjectHandle = ct.c_int()
    return c_GetObjectChild(clientID, parentObjectHandle, childIndex, ct.byref(childObjectHandle), operationMode), childObjectHandle.value

def simxGetObjectParent(clientID, childObjectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    parentObjectHandle = ct.c_int()
    return c_GetObjectParent(clientID, childObjectHandle, ct.byref(parentObjectHandle), operationMode), parentObjectHandle.value

def simxReadProximitySensor(clientID, sensorHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    detectionState = ct.c_ubyte()
    detectedObjectHandle = ct.c_int()
    detectedPoint  = (ct.c_float*3)()
    detectedSurfaceNormalVector = (ct.c_float*3)()
    ret = c_ReadProximitySensor(clientID, sensorHandle, ct.byref(detectionState), detectedPoint, ct.byref(detectedObjectHandle), detectedSurfaceNormalVector, operationMode)
    arr1 = []
    for i in xrange(3):
        arr1.append(detectedPoint[i])
    arr2 = []
    for i in xrange(3):
        arr2.append(detectedSurfaceNormalVector[i])
    return ret, bool(detectionState.value!=0), arr1, detectedObjectHandle.value, arr2

def simxLoadModel(clientID, modelPathAndName, options, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    baseHandle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(modelPathAndName) is unicode):
        modelPathAndName=modelPathAndName.encode(u'utf-8')
    return c_LoadModel(clientID, modelPathAndName, options, ct.byref(baseHandle), operationMode), baseHandle.value

def simxLoadUI(clientID, uiPathAndName, options, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    count = ct.c_int()
    uiHandles = ct.POINTER(ct.c_int)()
    if (sys.version_info[0] == 3) and (type(uiPathAndName) is unicode):
        uiPathAndName=uiPathAndName.encode(u'utf-8')
    ret = c_LoadUI(clientID, uiPathAndName, options, ct.byref(count), ct.byref(uiHandles), operationMode)

    handles = []
    if ret == 0:
        for i in xrange(count.value):
            handles.append(uiHandles[i])
        #free C buffers
        c_ReleaseBuffer(uiHandles)

    return ret, handles

def simxLoadScene(clientID, scenePathAndName, options, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(scenePathAndName) is unicode):
        scenePathAndName=scenePathAndName.encode(u'utf-8')
    return c_LoadScene(clientID, scenePathAndName, options, operationMode)

def simxStartSimulation(clientID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_StartSimulation(clientID, operationMode)

def simxPauseSimulation(clientID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_PauseSimulation(clientID, operationMode)

def simxStopSimulation(clientID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_StopSimulation(clientID, operationMode)

def simxGetUIHandle(clientID, uiName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    handle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(uiName) is unicode):
        uiName=uiName.encode(u'utf-8')
    return c_GetUIHandle(clientID, uiName, ct.byref(handle), operationMode), handle.value

def simxGetUISlider(clientID, uiHandle, uiButtonID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    position = ct.c_int()
    return c_GetUISlider(clientID, uiHandle, uiButtonID, ct.byref(position), operationMode), position.value

def simxSetUISlider(clientID, uiHandle, uiButtonID, position, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetUISlider(clientID, uiHandle, uiButtonID, position, operationMode)

def simxGetUIEventButton(clientID, uiHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    uiEventButtonID = ct.c_int()
    auxValues = (ct.c_int*2)()
    ret = c_GetUIEventButton(clientID, uiHandle, ct.byref(uiEventButtonID), auxValues, operationMode)
    arr = []
    for i in xrange(2):
        arr.append(auxValues[i])
    return ret, uiEventButtonID.value, arr

def simxGetUIButtonProperty(clientID, uiHandle, uiButtonID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    prop = ct.c_int()
    return c_GetUIButtonProperty(clientID, uiHandle, uiButtonID, ct.byref(prop), operationMode), prop.value

def simxSetUIButtonProperty(clientID, uiHandle, uiButtonID, prop, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetUIButtonProperty(clientID, uiHandle, uiButtonID, prop, operationMode)

def simxAddStatusbarMessage(clientID, message, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(message) is unicode):
        message=message.encode(u'utf-8')
    return c_AddStatusbarMessage(clientID, message, operationMode)

def simxAuxiliaryConsoleOpen(clientID, title, maxLines, mode, position, size, textColor, backgroundColor, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    consoleHandle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(title) is unicode):
        title=title.encode(u'utf-8')
    if position != None:
        c_position = (ct.c_int*2)(*position)
    else:
        c_position = None
    if size != None:
        c_size = (ct.c_int*2)(*size)
    else:
        c_size = None
    if textColor != None:
        c_textColor = (ct.c_float*3)(*textColor)
    else:
        c_textColor = None
    if backgroundColor != None:
        c_backgroundColor = (ct.c_float*3)(*backgroundColor)
    else:
        c_backgroundColor = None
    return c_AuxiliaryConsoleOpen(clientID, title, maxLines, mode, c_position, c_size, c_textColor, c_backgroundColor, ct.byref(consoleHandle), operationMode), consoleHandle.value

def simxAuxiliaryConsoleClose(clientID, consoleHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_AuxiliaryConsoleClose(clientID, consoleHandle, operationMode)

def simxAuxiliaryConsolePrint(clientID, consoleHandle, txt, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(txt) is unicode):
        txt=txt.encode(u'utf-8')
    return c_AuxiliaryConsolePrint(clientID, consoleHandle, txt, operationMode)

def simxAuxiliaryConsoleShow(clientID, consoleHandle, showState, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_AuxiliaryConsoleShow(clientID, consoleHandle, showState, operationMode)

def simxGetObjectOrientation(clientID, objectHandle, relativeToObjectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    eulerAngles = (ct.c_float*3)()
    ret = c_GetObjectOrientation(clientID, objectHandle, relativeToObjectHandle, eulerAngles, operationMode)
    arr = []
    for i in xrange(3):
        arr.append(eulerAngles[i])
    return ret, arr

def simxGetObjectQuaternion(clientID, objectHandle, relativeToObjectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    quaternion = (ct.c_float*4)()
    ret = c_GetObjectQuaternion(clientID, objectHandle, relativeToObjectHandle, quaternion, operationMode)
    arr = []
    for i in xrange(4):
        arr.append(quaternion[i])
    return ret, arr

def simxGetObjectPosition(clientID, objectHandle, relativeToObjectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    position = (ct.c_float*3)()
    ret = c_GetObjectPosition(clientID, objectHandle, relativeToObjectHandle, position, operationMode)
    arr = []
    for i in xrange(3):
        arr.append(position[i])
    return ret, arr

def simxSetObjectOrientation(clientID, objectHandle, relativeToObjectHandle, eulerAngles, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    angles = (ct.c_float*3)(*eulerAngles)
    return c_SetObjectOrientation(clientID, objectHandle, relativeToObjectHandle, angles, operationMode)

def simxSetObjectQuaternion(clientID, objectHandle, relativeToObjectHandle, quaternion, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    quat = (ct.c_float*4)(*quaternion)
    return c_SetObjectQuaternion(clientID, objectHandle, relativeToObjectHandle, quat, operationMode)

def simxSetObjectPosition(clientID, objectHandle, relativeToObjectHandle, position, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    c_position = (ct.c_float*3)(*position)
    return c_SetObjectPosition(clientID, objectHandle, relativeToObjectHandle, c_position, operationMode)

def simxSetObjectParent(clientID, objectHandle, parentObject, keepInPlace, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetObjectParent(clientID, objectHandle, parentObject, keepInPlace, operationMode)

def simxSetUIButtonLabel(clientID, uiHandle, uiButtonID, upStateLabel, downStateLabel, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if sys.version_info[0] == 3:
        if type(upStateLabel) is unicode:
            upStateLabel=upStateLabel.encode(u'utf-8')
        if type(downStateLabel) is unicode:
            downStateLabel=downStateLabel.encode(u'utf-8')
    return c_SetUIButtonLabel(clientID, uiHandle, uiButtonID, upStateLabel, downStateLabel, operationMode)

def simxGetLastErrors(clientID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    errors =[]
    errorCnt = ct.c_int()
    errorStrings = ct.POINTER(ct.c_char)()
    ret = c_GetLastErrors(clientID, ct.byref(errorCnt), ct.byref(errorStrings), operationMode)
    if ret == 0:
        s = 0
        for i in xrange(errorCnt.value):
            a = bytearray()
            while errorStrings[s] != '\0':
                if sys.version_info[0] == 3:
                    a.append(int.from_bytes(errorStrings[s],u'big'))
                else:
                    a.append(errorStrings[s])
                s += 1
            s += 1 #skip null
            if sys.version_info[0] == 3:
                errors.append(unicode(a,u'utf-8'))
            else:
                errors.append(unicode(a))

    return ret, errors

def simxGetArrayParameter(clientID, paramIdentifier, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    paramValues = (ct.c_float*3)()
    ret = c_GetArrayParameter(clientID, paramIdentifier, paramValues, operationMode)
    arr = []
    for i in xrange(3):
        arr.append(paramValues[i])
    return ret, arr

def simxSetArrayParameter(clientID, paramIdentifier, paramValues, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    c_paramValues = (ct.c_float*3)(*paramValues)
    return c_SetArrayParameter(clientID, paramIdentifier, c_paramValues, operationMode)

def simxGetBooleanParameter(clientID, paramIdentifier, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    paramValue = ct.c_ubyte()
    return c_GetBooleanParameter(clientID, paramIdentifier, ct.byref(paramValue), operationMode), bool(paramValue.value!=0)

def simxSetBooleanParameter(clientID, paramIdentifier, paramValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetBooleanParameter(clientID, paramIdentifier, paramValue, operationMode)

def simxGetIntegerParameter(clientID, paramIdentifier, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    paramValue = ct.c_int()
    return c_GetIntegerParameter(clientID, paramIdentifier, ct.byref(paramValue), operationMode), paramValue.value

def simxSetIntegerParameter(clientID, paramIdentifier, paramValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetIntegerParameter(clientID, paramIdentifier, paramValue, operationMode)

def simxGetFloatingParameter(clientID, paramIdentifier, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    paramValue = ct.c_float()
    return c_GetFloatingParameter(clientID, paramIdentifier, ct.byref(paramValue), operationMode), paramValue.value

def simxSetFloatingParameter(clientID, paramIdentifier, paramValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetFloatingParameter(clientID, paramIdentifier, paramValue, operationMode)

def simxGetStringParameter(clientID, paramIdentifier, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    paramValue = ct.POINTER(ct.c_char)()
    ret = c_GetStringParameter(clientID, paramIdentifier, ct.byref(paramValue), operationMode)

    a = bytearray()
    if ret == 0:
        i = 0
        while paramValue[i] != '\0':
            if sys.version_info[0] == 3:
                a.append(int.from_bytes(paramValue[i],u'big'))
            else:
                a.append(paramValue[i])
            i=i+1
    if sys.version_info[0] == 3:
        a=unicode(a,u'utf-8')
    else:
        a=unicode(a)
    return ret, a

def simxGetCollisionHandle(clientID, collisionObjectName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    handle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(collisionObjectName) is unicode):
        collisionObjectName=collisionObjectName.encode(u'utf-8')
    return c_GetCollisionHandle(clientID, collisionObjectName, ct.byref(handle), operationMode), handle.value

def simxGetCollectionHandle(clientID, collectionName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    handle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(collectionName) is unicode):
        collectionName=collectionName.encode(u'utf-8')
    return c_GetCollectionHandle(clientID, collectionName, ct.byref(handle), operationMode), handle.value

def simxGetDistanceHandle(clientID, distanceObjectName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    handle = ct.c_int()
    if (sys.version_info[0] == 3) and (type(distanceObjectName) is unicode):
        distanceObjectName=distanceObjectName.encode(u'utf-8')
    return c_GetDistanceHandle(clientID, distanceObjectName, ct.byref(handle), operationMode), handle.value

def simxReadCollision(clientID, collisionObjectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    collisionState = ct.c_ubyte()
    return c_ReadCollision(clientID, collisionObjectHandle, ct.byref(collisionState), operationMode), bool(collisionState.value!=0)

def simxReadDistance(clientID, distanceObjectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    minimumDistance = ct.c_float()
    return c_ReadDistance(clientID, distanceObjectHandle, ct.byref(minimumDistance), operationMode), minimumDistance.value

def simxRemoveObject(clientID, objectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_RemoveObject(clientID, objectHandle, operationMode)

def simxRemoveModel(clientID, objectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_RemoveModel(clientID, objectHandle, operationMode)

def simxRemoveUI(clientID, uiHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_RemoveUI(clientID, uiHandle, operationMode)

def simxCloseScene(clientID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_CloseScene(clientID, operationMode)

def simxGetObjects(clientID, objectType, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    objectCount = ct.c_int()
    objectHandles = ct.POINTER(ct.c_int)()

    ret = c_GetObjects(clientID, objectType, ct.byref(objectCount), ct.byref(objectHandles), operationMode)
    handles = []
    if ret == 0:
        for i in xrange(objectCount.value):
            handles.append(objectHandles[i])

    return ret, handles


def simxDisplayDialog(clientID, titleText, mainText, dialogType, initialText, titleColors, dialogColors, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    if titleColors != None:
        c_titleColors  = (ct.c_float*6)(*titleColors)
    else:
        c_titleColors  = None
    if dialogColors != None:
        c_dialogColors  = (ct.c_float*6)(*dialogColors)
    else:
        c_dialogColors  = None

    c_dialogHandle = ct.c_int()
    c_uiHandle = ct.c_int()
    if sys.version_info[0] == 3:
        if type(titleText) is unicode:
            titleText=titleText.encode(u'utf-8')
        if type(mainText) is unicode:
            mainText=mainText.encode(u'utf-8')
        if type(initialText) is unicode:
            initialText=initialText.encode(u'utf-8')
    return c_DisplayDialog(clientID, titleText, mainText, dialogType, initialText, c_titleColors, c_dialogColors, ct.byref(c_dialogHandle), ct.byref(c_uiHandle), operationMode), c_dialogHandle.value, c_uiHandle.value

def simxEndDialog(clientID, dialogHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_EndDialog(clientID, dialogHandle, operationMode)

def simxGetDialogInput(clientID, dialogHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    inputText = ct.POINTER(ct.c_char)()
    ret = c_GetDialogInput(clientID, dialogHandle, ct.byref(inputText), operationMode)

    a = bytearray()
    if ret == 0:
        i = 0
        while inputText[i] != '\0':
            if sys.version_info[0] == 3:
                a.append(int.from_bytes(inputText[i],u'big'))
            else:
                a.append(inputText[i])
            i = i+1

    if sys.version_info[0] == 3:
        a=unicode(a,u'utf-8')
    else:
        a=unicode(a)
    return ret, a


def simxGetDialogResult(clientID, dialogHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    result = ct.c_int()
    return c_GetDialogResult(clientID, dialogHandle, ct.byref(result), operationMode), result.value

def simxCopyPasteObjects(clientID, objectHandles, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    c_objectHandles  = (ct.c_int*len(objectHandles))(*objectHandles)
    c_objectHandles = ct.cast(c_objectHandles,ct.POINTER(ct.c_int)) # IronPython needs this
    newObjectCount   = ct.c_int()
    newObjectHandles = ct.POINTER(ct.c_int)()
    ret = c_CopyPasteObjects(clientID, c_objectHandles, len(objectHandles), ct.byref(newObjectHandles), ct.byref(newObjectCount), operationMode)

    newobj = []
    if ret == 0:
        for i in xrange(newObjectCount.value):
            newobj.append(newObjectHandles[i])

    return ret, newobj


def simxGetObjectSelection(clientID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    objectCount   = ct.c_int()
    objectHandles = ct.POINTER(ct.c_int)()
    ret = c_GetObjectSelection(clientID, ct.byref(objectHandles), ct.byref(objectCount), operationMode)

    newobj = []
    if ret == 0:
        for i in xrange(objectCount.value):
            newobj.append(objectHandles[i])

    return ret, newobj



def simxSetObjectSelection(clientID, objectHandles, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    c_objectHandles  = (ct.c_int*len(objectHandles))(*objectHandles)
    return c_SetObjectSelection(clientID, c_objectHandles, len(objectHandles), operationMode)

def simxClearFloatSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_ClearFloatSignal(clientID, signalName, operationMode)

def simxClearIntegerSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_ClearIntegerSignal(clientID, signalName, operationMode)

def simxClearStringSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_ClearStringSignal(clientID, signalName, operationMode)

def simxGetFloatSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    signalValue = ct.c_float()
    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_GetFloatSignal(clientID, signalName, ct.byref(signalValue), operationMode), signalValue.value

def simxGetIntegerSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    signalValue = ct.c_int()
    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_GetIntegerSignal(clientID, signalName, ct.byref(signalValue), operationMode), signalValue.value

def simxGetStringSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    signalLength = ct.c_int();
    signalValue = ct.POINTER(ct.c_ubyte)()
    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    ret = c_GetStringSignal(clientID, signalName, ct.byref(signalValue), ct.byref(signalLength), operationMode)

    a = bytearray()
    if ret == 0:
        for i in xrange(signalLength.value):
            a.append(signalValue[i])
    if sys.version_info[0] != 3:
        a=unicode(a)

    return ret, a

def simxGetAndClearStringSignal(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    signalLength = ct.c_int();
    signalValue = ct.POINTER(ct.c_ubyte)()
    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    ret = c_GetAndClearStringSignal(clientID, signalName, ct.byref(signalValue), ct.byref(signalLength), operationMode)

    a = bytearray()
    if ret == 0:
        for i in xrange(signalLength.value):
            a.append(signalValue[i])
    if sys.version_info[0] != 3:
        a=unicode(a)

    return ret, a

def simxReadStringStream(clientID, signalName, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    signalLength = ct.c_int();
    signalValue = ct.POINTER(ct.c_ubyte)()
    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    ret = c_ReadStringStream(clientID, signalName, ct.byref(signalValue), ct.byref(signalLength), operationMode)

    a = bytearray()
    if ret == 0:
        for i in xrange(signalLength.value):
            a.append(signalValue[i])
    if sys.version_info[0] != 3:
        a=unicode(a)

    return ret, a

def simxSetFloatSignal(clientID, signalName, signalValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_SetFloatSignal(clientID, signalName, signalValue, operationMode)

def simxSetIntegerSignal(clientID, signalName, signalValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(signalName) is unicode):
        signalName=signalName.encode(u'utf-8')
    return c_SetIntegerSignal(clientID, signalName, signalValue, operationMode)

def simxSetStringSignal(clientID, signalName, signalValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    sigV=signalValue
    if sys.version_info[0] == 3:
        if type(signalName) is unicode:
            signalName=signalName.encode(u'utf-8')
        if type(signalValue) is bytearray:
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=signalValue.encode(u'utf-8')
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
    else:
        if type(signalValue) is bytearray:
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=bytearray(signalValue)
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
    sigV=ct.cast(sigV,ct.POINTER(ct.c_ubyte)) # IronPython needs this
    return c_SetStringSignal(clientID, signalName, sigV, len(signalValue), operationMode)

def simxAppendStringSignal(clientID, signalName, signalValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    sigV=signalValue
    if sys.version_info[0] == 3:
        if type(signalName) is unicode:
            signalName=signalName.encode(u'utf-8')
        if type(signalValue) is bytearray:
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=signalValue.encode(u'utf-8')
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
    else:
        if type(signalValue) is bytearray:
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=bytearray(signalValue)
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
    sigV=ct.cast(sigV,ct.POINTER(ct.c_ubyte)) # IronPython needs this
    return c_AppendStringSignal(clientID, signalName, sigV, len(signalValue), operationMode)

def simxWriteStringStream(clientID, signalName, signalValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    sigV=signalValue
    if sys.version_info[0] == 3:
        if type(signalName) is unicode:
            signalName=signalName.encode(u'utf-8')
        if type(signalValue) is bytearray:
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=signalValue.encode(u'utf-8')
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
    else:
        if type(signalValue) is bytearray:
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=bytearray(signalValue)
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
    sigV=ct.cast(sigV,ct.POINTER(ct.c_ubyte)) # IronPython needs this
    return c_WriteStringStream(clientID, signalName, sigV, len(signalValue), operationMode)

def simxGetObjectFloatParameter(clientID, objectHandle, parameterID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    parameterValue = ct.c_float()
    return c_GetObjectFloatParameter(clientID, objectHandle, parameterID, ct.byref(parameterValue), operationMode), parameterValue.value

def simxSetObjectFloatParameter(clientID, objectHandle, parameterID, parameterValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetObjectFloatParameter(clientID, objectHandle, parameterID, parameterValue, operationMode)

def simxGetObjectIntParameter(clientID, objectHandle, parameterID, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    parameterValue = ct.c_int()
    return c_GetObjectIntParameter(clientID, objectHandle, parameterID, ct.byref(parameterValue), operationMode), parameterValue.value

def simxSetObjectIntParameter(clientID, objectHandle, parameterID, parameterValue, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetObjectIntParameter(clientID, objectHandle, parameterID, parameterValue, operationMode)

def simxGetModelProperty(clientID, objectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    prop = ct.c_int()
    return c_GetModelProperty(clientID, objectHandle, ct.byref(prop), operationMode), prop.value

def simxSetModelProperty(clientID, objectHandle, prop, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SetModelProperty(clientID, objectHandle, prop, operationMode)

def simxStart(connectionAddress, connectionPort, waitUntilConnected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(connectionAddress) is unicode):
        connectionAddress=connectionAddress.encode(u'utf-8')
    return c_Start(connectionAddress, connectionPort, waitUntilConnected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs)

def simxFinish(clientID):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_Finish(clientID)

def simxGetPingTime(clientID):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    pingTime = ct.c_int()
    return c_GetPingTime(clientID, ct.byref(pingTime)), pingTime.value

def simxGetLastCmdTime(clientID):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_GetLastCmdTime(clientID)

def simxSynchronousTrigger(clientID):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_SynchronousTrigger(clientID)

def simxSynchronous(clientID, enable):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_Synchronous(clientID, enable)

def simxPauseCommunication(clientID, enable):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_PauseCommunication(clientID, enable)

def simxGetInMessageInfo(clientID, infoType):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    info = ct.c_int()
    return c_GetInMessageInfo(clientID, infoType, ct.byref(info)), info.value

def simxGetOutMessageInfo(clientID, infoType):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    info = ct.c_int()
    return c_GetOutMessageInfo(clientID, infoType, ct.byref(info)), info.value

def simxGetConnectionId(clientID):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_GetConnectionId(clientID)

def simxCreateBuffer(bufferSize):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_CreateBuffer(bufferSize)

def simxReleaseBuffer(buffer):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    return c_ReleaseBuffer(buffer)

def simxTransferFile(clientID, filePathAndName, fileName_serverSide, timeOut, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(filePathAndName) is unicode):
        filePathAndName=filePathAndName.encode(u'utf-8')
    return c_TransferFile(clientID, filePathAndName, fileName_serverSide, timeOut, operationMode)

def simxEraseFile(clientID, fileName_serverSide, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if (sys.version_info[0] == 3) and (type(fileName_serverSide) is unicode):
        fileName_serverSide=fileName_serverSide.encode(u'utf-8')
    return c_EraseFile(clientID, fileName_serverSide, operationMode)

def simxCreateDummy(clientID, size, color, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    handle = ct.c_int()
    if color != None:
        c_color = (ct.c_ubyte*12)(*color)
    else:
        c_color = None
    return c_CreateDummy(clientID, size, c_color, ct.byref(handle), operationMode), handle.value

def simxQuery(clientID, signalName, signalValue, retSignalName, timeOutInMs):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    retSignalLength = ct.c_int();
    retSignalValue = ct.POINTER(ct.c_ubyte)()

    sigV=signalValue
    if sys.version_info[0] == 3:
        if type(signalName) is unicode:
            signalName=signalName.encode(u'utf-8')
        if type(retSignalName) is unicode:
            retSignalName=retSignalName.encode(u'utf-8')
        if type(signalValue) is bytearray:
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=signalValue.encode(u'utf-8')
            sigV  = (ct.c_ubyte*len(signalValue))(*signalValue)
    else:
        if type(signalValue) is bytearray:
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
        if type(signalValue) is unicode:
            signalValue=bytearray(signalValue)
            sigV = (ct.c_ubyte*len(signalValue))(*signalValue)
    sigV=ct.cast(sigV,ct.POINTER(ct.c_ubyte)) # IronPython needs this

    ret = c_Query(clientID, signalName, sigV, len(signalValue), retSignalName, ct.byref(retSignalValue), ct.byref(retSignalLength), timeOutInMs)

    a = bytearray()
    if ret == 0:
        for i in xrange(retSignalLength.value):
            a.append(retSignalValue[i])
    if sys.version_info[0] != 3:
        a=unicode(a)

    return ret, a

def simxGetObjectGroupData(clientID, objectType, dataType, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    handles =[]
    intData =[]
    floatData =[]
    stringData =[]
    handlesC = ct.c_int()
    handlesP = ct.POINTER(ct.c_int)()
    intDataC = ct.c_int()
    intDataP = ct.POINTER(ct.c_int)()
    floatDataC = ct.c_int()
    floatDataP = ct.POINTER(ct.c_float)()
    stringDataC = ct.c_int()
    stringDataP = ct.POINTER(ct.c_char)()
    ret = c_GetObjectGroupData(clientID, objectType, dataType, ct.byref(handlesC), ct.byref(handlesP), ct.byref(intDataC), ct.byref(intDataP), ct.byref(floatDataC), ct.byref(floatDataP), ct.byref(stringDataC), ct.byref(stringDataP), operationMode)

    if ret == 0:
        for i in xrange(handlesC.value):
            handles.append(handlesP[i])
        for i in xrange(intDataC.value):
            intData.append(intDataP[i])
        for i in xrange(floatDataC.value):
            floatData.append(floatDataP[i])
        s = 0
        for i in xrange(stringDataC.value):
            a = bytearray()
            while stringDataP[s] != '\0':
                if sys.version_info[0] == 3:
                    a.append(int.from_bytes(stringDataP[s],u'big'))
                else:
                    a.append(stringDataP[s])
                s += 1
            s += 1 #skip null
            if sys.version_info[0] == 3:
                a=unicode(a,u'utf-8')
            else:
                a=unicode(a)
            stringData.append(a)

    return ret, handles, intData, floatData, stringData

def simxCallScriptFunction(clientID, scriptDescription, options, functionName, inputInts, inputFloats, inputStrings, inputBuffer, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    inputBufferV=inputBuffer
    if sys.version_info[0] == 3:
        if type(scriptDescription) is unicode:
            scriptDescription=scriptDescription.encode(u'utf-8')
        if type(functionName) is unicode:
            functionName=functionName.encode(u'utf-8')
        if type(inputBuffer) is bytearray:
            inputBufferV  = (ct.c_ubyte*len(inputBuffer))(*inputBuffer)
        if type(inputBuffer) is unicode:
            inputBuffer=inputBuffer.encode(u'utf-8')
            inputBufferV  = (ct.c_ubyte*len(inputBuffer))(*inputBuffer)
    else:
        if type(inputBuffer) is bytearray:
            inputBufferV = (ct.c_ubyte*len(inputBuffer))(*inputBuffer)
        if type(inputBuffer) is unicode:
            inputBuffer=bytearray(inputBuffer)
            inputBufferV = (ct.c_ubyte*len(inputBuffer))(*inputBuffer)
    inputBufferV=ct.cast(inputBufferV,ct.POINTER(ct.c_ubyte)) # IronPython needs this

    c_inInts  = (ct.c_int*len(inputInts))(*inputInts)
    c_inInts = ct.cast(c_inInts,ct.POINTER(ct.c_int)) # IronPython needs this
    c_inFloats  = (ct.c_float*len(inputFloats))(*inputFloats)
    c_inFloats = ct.cast(c_inFloats,ct.POINTER(ct.c_float)) # IronPython needs this

    concatStr=u''.encode(u'utf-8')
    for i in xrange(len(inputStrings)):
        a=inputStrings[i]
        a=a+u'\0'
        if type(a) is unicode:
            a=a.encode(u'utf-8')
        concatStr=concatStr+a
    c_inStrings  = (ct.c_char*len(concatStr))(*concatStr)

    intDataOut =[]
    floatDataOut =[]
    stringDataOut =[]
    bufferOut =bytearray()

    intDataC = ct.c_int()
    intDataP = ct.POINTER(ct.c_int)()
    floatDataC = ct.c_int()
    floatDataP = ct.POINTER(ct.c_float)()
    stringDataC = ct.c_int()
    stringDataP = ct.POINTER(ct.c_char)()
    bufferS = ct.c_int()
    bufferP = ct.POINTER(ct.c_ubyte)()

    ret = c_CallScriptFunction(clientID,scriptDescription,options,functionName,len(inputInts),c_inInts,len(inputFloats),c_inFloats,len(inputStrings),c_inStrings,len(inputBuffer),inputBufferV,ct.byref(intDataC),ct.byref(intDataP),ct.byref(floatDataC),ct.byref(floatDataP),ct.byref(stringDataC),ct.byref(stringDataP),ct.byref(bufferS),ct.byref(bufferP),operationMode)

    if ret == 0:
        for i in xrange(intDataC.value):
            intDataOut.append(intDataP[i])
        for i in xrange(floatDataC.value):
            floatDataOut.append(floatDataP[i])
        s = 0
        for i in xrange(stringDataC.value):
            a = bytearray()
            while stringDataP[s] != '\0':
                if sys.version_info[0] == 3:
                    a.append(int.from_bytes(stringDataP[s],u'big'))
                else:
                    a.append(stringDataP[s])
                s += 1
            s += 1 #skip null
            if sys.version_info[0] == 3:
                a=unicode(a,u'utf-8')
            else:
                a=unicode(a)
            stringDataOut.append(a)
        for i in xrange(bufferS.value):
            bufferOut.append(bufferP[i])
    if sys.version_info[0] != 3:
        bufferOut=unicode(bufferOut)

    return ret, intDataOut, floatDataOut, stringDataOut, bufferOut

def simxGetObjectVelocity(clientID, objectHandle, operationMode):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    linearVel  = (ct.c_float*3)()
    angularVel = (ct.c_float*3)()
    ret = c_GetObjectVelocity(clientID, objectHandle, linearVel, angularVel, operationMode)
    arr1 = []
    for i in xrange(3):
        arr1.append(linearVel[i])
    arr2 = []
    for i in xrange(3):
        arr2.append(angularVel[i])
    return ret, arr1, arr2

def simxPackInts(intList):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if sys.version_info[0] == 3:
        s=str()
        for i in xrange(len(intList)):
            s=s+struct.pack(u'<i',intList[i])
        s=bytearray(s)
    else:
        s=u''
        for i in xrange(len(intList)):
            s+=struct.pack(u'<i',intList[i])
    return s

def simxUnpackInts(intsPackedInString):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    b=[]
    for i in xrange(int(len(intsPackedInString)/4)):
        b.append(struct.unpack(u'<i',intsPackedInString[4*i:4*(i+1)])[0])
    return b

def simxPackFloats(floatList):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''

    if sys.version_info[0] == 3:
        s=str()
        for i in xrange(len(floatList)):
            s=s+struct.pack(u'<f',floatList[i])
        s=bytearray(s)
    else:
        s=u''
        for i in xrange(len(floatList)):
            s+=struct.pack(u'<f',floatList[i])
    return s

def simxUnpackFloats(floatsPackedInString):
    u'''
    Please have a look at the function description/documentation in the V-REP user manual
    '''
    b=[]
    for i in xrange(int(len(floatsPackedInString)/4)):
        b.append(struct.unpack(u'<f',floatsPackedInString[4*i:4*(i+1)])[0])
    return b
