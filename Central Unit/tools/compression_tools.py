__author__ = 'olivier'

import numpy as np
from struct import pack,unpack, calcsize
import bz2
from skimage.io import imsave, imread
import cv2
#from scipy.misc import imsave,imread

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

# may not work due to a missing plugin...
def np_rgb_to_payload(rgb):
    # convert 3D RGB uint8 np.array into JPG compressed byterarray
    assert(rgb.ndim == 3)
    assert(rgb.dtype == np.uint8)
    s = BytesIO()
    imsave(s,rgb,'jpg')
    s.seek(0)
    payload = bytearray(s.read())
    print(len(payload))
    print(payload[:10])
    return payload

def np_to_payload(arr):
    # convert 2D/3D uint8 to byterarray
    # payload = m(uint16),n(uint16),p(uint16) bytes
    assert(1 < arr.ndim < 4)
    assert(arr.dtype == np.uint8)
    if arr.ndim == 2:
        m,n = arr.shape
        p = 0
    else:
        m,n,p = arr.shape
    payload = bytearray(pack('hhh',m,n,p)) + bytearray(arr.tostring())
    return payload


def ts_bb_signarray_to_payload(ts, bb, arr):
    '''convert triple of (timestamp, boundingbox(x0,y0,w,h), arr) with arr
       a 2D/3D uint8 to byterarray to a binary MQTT payload
      payload = ts(double), x0(uint16), y0(uint16), w(uint16), h(uint16),
                m(uint16),n(uint16),p(uint16) bytes'''
    x0, y0, w, h = bb
    assert(1 < arr.ndim < 4)
    assert(arr.dtype == np.uint8)
    if arr.ndim == 2:
        m, n = arr.shape
        p = 0
    else:
        m, n, p = arr.shape
    fmt = 'dhhhhhhh'
    payload = bytearray(pack(fmt, ts, x0, y0, w, h, m, n, p))
    payload = payload + bytearray(arr.tostring())
    return payload


def payload_to_ts_bb_signarray(payload):
    ''''convert a binary MQTT payload containing a triple of
      (timestamp, boundingbox(x0,y0,w,h), arr) with arr
      a 2D/3D uint8 backinto such a tuple'''
    fmt = 'dhhhhhhh'
    i = calcsize(fmt)
    ts, x0, y0, w, h, m, n, p = unpack(fmt, payload[:i])
    if p == 0:
        arr = np.fromstring(payload[i:], dtype=np.uint8).reshape(m, n)
    else:
        arr = np.fromstring(payload[i:], dtype=np.uint8).reshape(m, n, p)
    return (ts, (x0, y0, w, h), arr)


def ts_ref_np_to_payload(ts, ref, arr):
    # convert 2D/3D uint8 to byterarray
    # payload = m(uint16),n(uint16),p(uint16) bytes
    assert(1 < arr.ndim < 4)
    assert(arr.dtype == np.uint8)
    if arr.ndim == 2:
        m,n = arr.shape
        p = 0
    else:
        m,n,p = arr.shape
    payload = bytearray(pack('ddhhh',ts, ref, m,n,p)) + bytearray(arr.tostring())
    return payload

def payload_to_ts_ref_np(payload):
    fmt = 'ddhhh'
    i = calcsize(fmt)
    # convert payload to 2D/3D numpy array (uint8)
    ts,ref,m,n,p = unpack(fmt,payload[:i]) # retrieve array size from 3 first uint16
    if p == 0:
        data = np.fromstring(payload[i:],dtype=np.uint8).reshape(m,n)
    else:
        data = np.fromstring(payload[i:],dtype=np.uint8).reshape(m,n,p)
    return (ts, ref, data)




# packbit+compression
def compress_bool_array(arr):
    # returns a gz2 compressed buffer
    pb = np.packbits(arr.astype(np.uint8))
    compressed = bz2.compress(pb)
    # print('raw:',len(arr.flatten()),arr.shape)
    # print('pb:',len(pb))
    # print('bz2',len(compressed))
    return compressed

def decompress_bool_array(carr,shape):
    # returns a 2d bool array from compressed buffer
    ba = bytearray(bz2.decompress(carr))
    upb = np.unpackbits(ba)[:shape[0]*shape[1]] # if non 8 multiple
    return upb.reshape(shape)

def ts_mask_to_payload(ts, arr):
    # convert 2D bool to byterarray; prefixed with timestamp
    # uses compression
    # payload = m(uint16),n(uint16),bytes
    assert(arr.ndim == 2)
    assert(arr.dtype == np.bool)
    m,n = arr.shape
    payload = bytearray(pack('dhh',ts,m,n))+bytearray(compress_bool_array(arr))
    return payload

def payload_to_ts_mask(payload):
    fmt = 'dhh'
    i = calcsize(fmt)
    # convert timestamped and compressed 2D bool array to numpy array
    ts, md,nd = unpack(fmt,payload[:i]) # retrieve array size from 1st float and subsequen 2 first uint16
    data = decompress_bool_array(payload[i:],(md,nd))
    return (ts, data)


def bool_to_payload(arr):
    # convert 2D bool to byterarray
    # uses compression
    # payload = m(uint16),n(uint16),bytes
    assert(arr.ndim == 2)
    assert(arr.dtype == np.bool)
    m,n = arr.shape
    payload = bytearray(pack('hh',m,n))+bytearray(compress_bool_array(arr))
    return payload



def payload_to_bool(payload):
    # convert compressed 2D bool array to numpy array
    md,nd = unpack('hh',payload[:4]) # retrieve array size from 2 first uint16
    data = decompress_bool_array(payload[4:],(md,nd))
    return data

def payload_to_np(payload):
    # convert payload to 2D/3D numpy array (uint8)
    m,n,p = unpack('hhh',payload[:6]) # retrieve array size from 3 first uint16
    if p == 0:
        data = np.fromstring(payload[6:],dtype=np.uint8).reshape(m,n)
    else:
        data = np.fromstring(payload[6:],dtype=np.uint8).reshape(m,n,p)
    return data

def payload_to_np_rgb(payload):
    # convert payload into a 3D unit8 RGB image (expected to be JPG compressed)
    s = BytesIO()
    s.write(payload)
    rgb = imread(s)
    return rgb

def ts_ref_rgb_to_payload(ts, ref, rgb):
    # convert timestamp, ref timestamp + 3D RGB uint8 np.array into JPG compressed byterarray
    assert(rgb.ndim == 3)
    assert(rgb.dtype == np.uint8)
    (succeeded, jpg) = cv2.imencode('.png', rgb)
    if not succeeded:
        logging.error("IMEncode failed!")
    payload = bytearray(pack('dd', ts, ref))+bytearray(jpg)
    return payload


def payload_to_ts_ref_rgb(payload):
    # convert payload to (timestamp, ref timestamp, rgb) where rgb is a 3D RGB uint8 np array
    # convert payload into a 3D unit8 RGB image (expected to be JPG compressed)    
    fmt = 'dd'
    i = calcsize(fmt)
    # convert timestamped and compressed 2D bool array to numpy array
    ts, ref = unpack(fmt, payload[:i])
    rgb = cv2.imdecode(np.fromstring(payload[i:], dtype=np.uint8), 1) # last argument "1" means that we return a RGB image
    return (ts, ref, rgb)
