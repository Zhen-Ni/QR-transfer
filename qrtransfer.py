#!/usr/bin/env python3


"""
Transfer data using QR codes.

A single QR code can contain more than 2k data at most. Thus it is a good idea to
store information in pictures or video containing QR codes. This program encodes
data into QR codes and stores these QR codes in pictures or in a video.
Corresponding decoding preocedure is also provided to recover data.

The data is splited and stored in the QR codes. For each QR code, the first four
bytes store the index of the QR code, and the fourth to eighth bytes store the
total number of QR codes used. To make sure the QR code can be identified, base64
encoding is used for storing these data. As base64 encoding needs four bytes to
store three bytes information, the index and number of QR codes are actually three
bytes. The 8th-12th bytes are the parameters for forward erasure correction, 
which is also base64 encoded. The lower byte of the unencoded data contains m
and the last but two byte contains k. The descirbtion of k and m can be found
in python package `zfec`. The data are stored after the 12th byte, which is also
base64 encoded. The data stored in the qr codes needs to be processed first.
The first four bytes is an unsigned int indicating the length of the data to be
stored and additional fec data is appended at the end of the data.
Thus, the data structure for each QR code is as follows:

bytes[0:4]: base64_encode(`index`), where `index` is a 3-byte unsigned int
bytes[4:8]: base64_encode(`n`), where `n` is also 3-byte indicating the total
number of qr codes.
bytes[8:12]: base64_encode(`bytes(0,k,m)`), where k and m are fec parameters
bytes[12:]: base64_encode(`qrdata`), where qrdata is the processed data
"""

import os
import math
import base64
import numpy as np
import qrcode
from pyzbar import pyzbar
import cv2
import zfec
import argparse

from multiprocessing import pool

MAX_FEC_M = 256
SIZE_INDEX = 3
SIZE_PREFIX = 4
SIZE_DATASIZE = 4
BYTE_ORDER = 'big'
BOX_SIZE = 4
FRAMERATE = 15
BCH_POLYNOMIAL = 8219


def encode_index(n):
    b = n.to_bytes(SIZE_INDEX, BYTE_ORDER)
    b = base64.encodebytes(b)
    b = b''.join(b.split(b'\n'))
    return b


def decode_index(b):
    b = base64.decodebytes(b)
    n = b.hex()
    n = int(n, 16)
    return n


def encode_data(b):
    b = base64.encodebytes(b)
    b = b''.join(b.split(b'\n'))
    return b


def decode_data(b):
    b = base64.decodebytes(b)
    return b


def encode_fecinfo(k, m):
    n = (k << 8) + m
    return encode_index(n)


def decode_fecinfo(b):
    n = decode_index(b)
    k = n >> 8
    m = n & 255
    return k, m


class Encoder:
    def __init__(self, data, fec_ratio=0.1, version=35, qrerror='L'):
        """Encode data into QR codes.

        Parameters
        ----------
        data: bytes
            Data to be encoded into QR codes, can be arbitrary length.
        fec_ratio: float between 0 and 1
            Use addtional fec data to make the generated QR codes more robustic.
            fec_ratio is the ratio of erasure correction data and the total
            generated data.
        version: int between 1 and 40
            Version of the QR code to use.
        qrerror: {'L', 'M', 'Q', 'H'}
            Error correction level of the QR code.
        """
        self._version = version
        self._fec_ratio = fec_ratio
        qrerror_correction = {'L': qrcode.constants.ERROR_CORRECT_L,
                              'M': qrcode.constants.ERROR_CORRECT_M,
                              'Q': qrcode.constants.ERROR_CORRECT_Q,
                              'H': qrcode.constants.ERROR_CORRECT_H}[qrerror]
        self._qrerror_correction = qrerror_correction

        data = self._prepare_data(data)
        data_list = self._chunk_data(data, version, qrerror_correction)
        self._k, self._m = None, None
        data_list = self._fec_encode(data_list, fec_ratio)
        self._data_list = self._qrformat_encode(data_list)
        self._qrcodes = None

    def _prepare_data(self, data):
        if isinstance(data, str):
            data = data.encode()
        size = len(data)
        size = size.to_bytes(SIZE_DATASIZE, BYTE_ORDER)
        data = size + data
        return data        
        
    def _chunk_data(self, data, version, qrerror_correction):
        size = len(data)
        maxchunksize = qrcode.util.BIT_LIMIT_TABLE[qrerror_correction][version]
        size_dict = qrcode.util.mode_sizes_for_version(version)
        chunk_prefix_size = max(size_dict.values())
        chunk_prefix_size += 4  # need 4 bits to store mode (qrcode.main:159)
        maxchunksize = (maxchunksize - chunk_prefix_size) // 8
        maxchunksize -= SIZE_PREFIX * 3  # frame header
        maxchunksize = int(maxchunksize / 4) * 3  # use base64 encoding
        nchunks = size / maxchunksize
        nchunks = math.ceil(size/maxchunksize)

        data_list = []
        for i in range(nchunks):
            beg = i*maxchunksize
            end = beg + maxchunksize
            if end < size:
                datai = data[beg: end]
            else:
                datai = data[beg:] + b'\0' * (end - size)
            data_list.append(datai)
        return data_list

    def _fec_encode(self, data_list, fec_ratio):
        size = len(data_list)
        chunksize = len(data_list[0])
        blocksize = math.ceil(size / (1 - fec_ratio))
        nblocks = math.ceil(blocksize / MAX_FEC_M)
        blocksize = math.ceil(blocksize / nblocks)
        remainder = size % nblocks
        if remainder:
            data_list.extend([b'\0' * chunksize] * (nblocks - remainder))
        m = blocksize
        k = len(data_list) // nblocks
        self._k, self._m = k, m
        encoder = zfec.Encoder(k, m)
        fec_data_list_mapped = []
        for i in range(nblocks):
            fec_data = encoder.encode(data_list[i::nblocks])
            fec_data_list_mapped.append(fec_data)
        fec_data_list = []
        for i in range(blocksize):
            for j in range(nblocks):
                fec_data_list.append(fec_data_list_mapped[j][i])
        return fec_data_list
            
    def _qrformat_encode(self, data_list):
        new_list = []
        nframes = len(data_list)
        k, m = self._k, self._m
        for i, data in enumerate(data_list):
            code = self._encode(i, nframes, k, m, data)
            new_list.append(code)
        return new_list
    
    def _encode(self, index, nframes, k, m, data):
        header_index = encode_index(index)
        header_nframes = encode_index(nframes)
        header_fec = encode_fecinfo(k, m)
        header = header_index + header_nframes + header_fec
        code = header + encode_data(data)
        return code

    def _make_parallel_helper(self, data, version, error_correction, box_size):
        qr = qrcode.QRCode(version, error_correction,
                           box_size=box_size)
        qr.add_data(data, optimize=0)
        qr.make(fit=False)
        return qr

    def make_serial(self, box_size=BOX_SIZE):
        self._qrcodes = []
        for i, data in enumerate(self._data_list):
            qr = qrcode.QRCode(self._version, self._qrerror_correction,
                               box_size=box_size)
            qr.add_data(data, optimize=0)
            qr.make(fit=False)
            self._qrcodes.append(qr)
        
    def make_parallel(self, box_size=BOX_SIZE):
        self._qrcodes = []
        p = pool.Pool()
        results = []
        for i, data in enumerate(self._data_list):
            args = data, self._version, self._qrerror_correction, box_size
            r = p.apply_async(self._make_parallel_helper, args)
            results.append(r)
        for r in results:
            qr = r.get()
            self._qrcodes.append(qr)

    make = make_serial

    def save_images(self, folder):
        if self._qrcodes is None:
            self.make()
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        for i, qr in enumerate(self._qrcodes):
            img = qr.make_image()
            img.save(os.path.join(folder, '{}.png'.format(i)), format='png')
            
    def save_video(self, filename, framerate=FRAMERATE):
        if self._qrcodes is None:
            self.make()
        images = []
        for i, qr in enumerate(self._qrcodes):
            img = qr.make_image().get_image()
            images.append(img)
        size = images[0].size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(filename, -1, framerate, size)
        for img in images:
            img = np.array(img, dtype=np.uint8) * 255
##            img = cv2.UMat(img)
            writer.write(img)
        writer.release()


class Decoder:
    def __init__(self):
        self._fecinfo = None
        self._data_list = None

    def _decode(self, code):
        index = None
        nframes = None
        data = None
        index = code.data[:SIZE_PREFIX]
        index = decode_index(index)
        nframes = code.data[SIZE_PREFIX:SIZE_PREFIX*2]
        nframes = decode_index(nframes)
        fecbits = code.data[SIZE_PREFIX*2:SIZE_PREFIX*3]
        fecinfo = decode_fecinfo(fecbits)
        data = code.data[SIZE_PREFIX*3:]
        data = decode_data(data)
        return index, nframes, fecinfo, data

    def _add_data(self, index, nframes, fecinfo, data):
        # Do nothing if this qrcode contains no data
        if nframes is None:
            return
        # Set self._data_list to list of Nones if it has not been
        # initialized, else check the whether the length of self._data_list
        # equals nframes.
        if self._data_list is None:
            self._data_list = [None] * nframes
            self._fecinfo = fecinfo
        else:
            if not nframes == len(self._data_list):
                raise ValueError('number of frames do not match')
            if not fecinfo == self._fecinfo:
                raise ValueError('k and m in fecinfo do not match')
        self._data_list[index] = data

    def _decode_image(self, image):
        """Decode the image and save its contents."""
        codes = pyzbar.decode(image)
        for code in codes:
            index, nframes, fecinfo, data = self._decode(code)
            self._add_data(index, nframes, fecinfo, data)
        
    def add_images(self, folder):
        files = os.listdir(folder)
        for file in files:
            if os.path.isdir(os.path.join(folder, file)):
                continue
            image = cv2.imread(os.path.join(folder, file))
            self._decode_image(image)

    def add_image(self, filename):
        image = cv2.imread(filename)
        self._decode_image(image)

    def add_video(self, filename):
        video = cv2.VideoCapture(filename)
        success = True
        while success:
            success, image = video.read()
            if success:
                self._decode_image(image)
        video.release()

    def _fec_decode(self, data_list):
        k, m = self._fecinfo
        nframes = len(data_list)
        nblocks = nframes // m
        decoder = zfec.Decoder(k, m)
        recovered = []
        for i in range(nblocks):
            packet = data_list[i::nblocks]
            blocks = []
            blocknums = []
            for j, d in enumerate(packet):
                if d is not None:
                    blocks.append(d)
                    blocknums.append(j)
            if len(blocks) < k:
                raise ValueError('fail to recover data')
            recovered.append(decoder.decode(blocks[:k], blocknums[:k]))
        data_list_recovered = []
        for i in range(k):
            for j in range(nblocks):
                data_list_recovered.append(recovered[j][i])
        return data_list_recovered

    def _prepare_data(self, data_list):
        data = b''.join(data_list)
        data_size = data[:SIZE_DATASIZE]
        data_size = int(data_size.hex(), 16)
        data = data[SIZE_DATASIZE:data_size+SIZE_DATASIZE]
        return data
    
    def get(self):
        if self._data_list is None:
            return None
        data_list = self._fec_decode(self._data_list)
        data = self._prepare_data(data_list)
        return data
        
        
def _test():
    with open('test.txt', 'rb') as f:
        s = f.read()
    encoder = Encoder(s, version=35)
    encoder.save_images('test')
    encoder.save_video('test.mp4')

    decoder = Decoder()
    # decoder.add_images('test')
    decoder.add_video('test.mp4')
    s2 = decoder.get()
    with open('test2.txt', 'wb') as f:
        f.write(s2)


def _encode(source, target, fec_ratio, version, framerate):
    with open(source, 'rb') as f:
        data = f.read()
    # Encoder.make = Encoder.make_parallel
    encoder = Encoder(data, fec_ratio, version)
    encoder.make()
    encoder.save_video(target, framerate)
    return 0


def _decode(sources, target):
    decoder = Decoder()
    for source in sources:
        decoder.add_video(source)
    data = decoder.get()
    with open(target, 'wb') as f:
        f.write(data)
    return 0

def _main():
    parser = argparse.ArgumentParser(description="Encode a file into QR code"
                                     " video or decode a QR code video.")
    parser.add_argument('mode', choices=('encode', ('decode')),
                        help='select mode')
    parser.add_argument('-if', metavar='input', type=str, nargs='+',
                        help='input file name', required=True)
    parser.add_argument('-of', metavar='output', type=str,
                        help='output file name', required=True)
    parser.add_argument('-fec', metavar='ratio', type=float, default=0.1,
                        help='fec ratio')
    parser.add_argument('-version', metavar='n', type=int, default=35,
                        help='QR code version')
    parser.add_argument('-fps', type=int, default=FRAMERATE,
                        help='frame rate')
    args = parser.parse_args()
    source = getattr(args, 'if')
    target = args.of
    
    if args.mode == 'encode':
        if len(source) != 1:
            raise ValueError('can only encode one file once')
        _encode(source[0], target, args.fec, args.version, args.fps)
    else:
        _decode(source, target)
    return 0
        

if __name__ == '__main__':
    _main()
    
