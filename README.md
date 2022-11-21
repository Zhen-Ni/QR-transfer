# QR-transfer

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
- bytes[0:4]: base64_encode(`index`), where `index` is a 3-byte unsigned int
- bytes[4:8]: base64_encode(`n`), where `n` is also 3-byte indicating the total
number of qr codes.
- bytes[8:12]: base64_encode(`bytes(0,k,m)`), where k and m are fec parameters
- bytes[12:]: base64_encode(`qrdata`), where qrdata is the processed data
