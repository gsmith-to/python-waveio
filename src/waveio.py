
#
# WAVE file I/O
#
import struct
import numpy as np
import sys

"""
WAVE reader, supports 8-to-16-bit mono and stereo PCM data

 w= WaveFileReader('whatever.wav')

 (the filename can also be file open for binary read)

 w acts more or less like a read-only array object;
 its shape is (nn,) for mono, or (nn,2) for stereo.
 you can test its length, and read slices. Strides,
 and use of the second index for stereo wav, are
 not supported.

 The values returned are scaled so that they are
 always +/- 32K, regardless of the underlying sample
 format; and are returned as numpy int16 arrays.
 (except in the case of a single index e.g. w[100] from
 a mono file,  which returns an int).

 some read-only fields:
   nchans                1 or 2
   sample_rate           samples/sec
   sample_count          samples in the file
   duration              length of file in seconds (float)
   shape                 Either (sample_count,) or (sample_count,2)
"""
__all__ = ['RiffReader', 'WaveFileReader', 'WaveFileWriter']


Need_byte_swap = (sys.byteorder != 'little')

class RiffReader(object):
    "base class for RIFF reader"
    __slots__ = ['f', 'file_length', 'RIFFtype',
        'chunklist' # list of ('chkid', posn, length)
        ]
    #
    # in chunklist, 'posn' is for the data part of the chunk,
    # and 'length' is the data part only.
    #
    expected_RIFFtype = None

    def __init__(self, fin ):
        if hasattr(fin, 'read'):
            f = fin
        else:
            f = open(fin, 'rb')
        self.f = f
        hdr = f.read(12)

        s1,length,s2 = struct.unpack('<4sL4s',hdr)
        self.RIFFtype = s2
        exp_RIFFtype = self.expected_RIFFtype

        if (exp_RIFFtype is not None and
           exp_RIFFtype != s2):
            raise ValueError, "expecting RIFF type %r, got %r" %(
                exp_RIFFtype, s2 )

        self.file_length= length
        if s1 != 'RIFF':
            raise ValueError, "RIFF header not found"

        chunklist = []
        pos = 12
        nextpos = 12
        while nextpos < length:
            if nextpos != pos:
                if nextpos < pos:
                    raise ValueError, "bad chunk length"
                f.seek(nextpos)
                pos = nextpos
            chkhdr = f.read(8)
            pos += 8
            chktype, chklen = struct.unpack('<4sL',chkhdr)
            nextpos = pos + ((chklen + 1)&~1)
            chunklist.append( ( chktype, pos, chklen) )
        self.chunklist = chunklist

    def find_chunk_data(self, chktype):
        for p in self.chunklist:
            if p[0] == chktype:
                return p
        raise ValueError, "chunk type %r not found" % chktype

    def get_chunk_data(self, chktype):
        p = self.find_chunk_data(chktype)
        f = self.f
        f.seek( p[1])
        return f.read(p[2])



class WaveFileReader(RiffReader):
    __slots__= ['nchans', 
        'bits_per_samp',
         'bytes_per_frame',
         'sample_rate',
         'sample_count',
         'datapos',     # position in file
         'databytes',
         'duration',
         'samp_decoder',
         'arr_decoder',
         'shape'
         ]

    expected_RIFFtype = 'WAVE'

    def __init__(self, *parms, **kw):
        RiffReader.__init__(self, *parms, **kw)
        dat = self.get_chunk_data('fmt ')
        if len(dat)==16:
            dat += '\0\0'
        (comp_code, nchans, samp_rate, byte_per_sec,
            blk_align, bits_per_samp, extra_bytes) = struct.unpack('<HHLL3H', dat[:18])
        self.nchans = nchans
        if comp_code != 1 or nchans not in (1,2) or not ( 8 <= bits_per_samp <= 16):
            raise ValueError, "unsupported RIFF format"
        bytes_per_frame = nchans*( 1 + (bits_per_samp>8) )
        self.bytes_per_frame = bytes_per_frame
        self.bits_per_samp = bits_per_samp
        self.sample_rate = samp_rate

        #
        # find data
        #
        tmp, datapos, databytes = self.find_chunk_data( 'data')
        self.datapos = datapos
        self.databytes = databytes
        self.sample_count = int(databytes//bytes_per_frame)
        self.duration = self.sample_count / float( samp_rate)
        #
        # arrange decoder
        #
        if bits_per_samp == 8:
            if nchans == 2:
                sdec = dec_8_stereo
                adec = dec_8_stereo_arr
            else:
                sdec = dec_8_mono
                adec = dec_8_mono_arr
        elif bits_per_samp == 16:
            if nchans == 2:
                sdec = dec_16_stereo
                adec = dec_16_stereo_arr
            else:
                sdec = dec_16_mono
                adec = dec_16_mono_arr
        self.samp_decoder = sdec
        self.arr_decoder = adec
        if nchans == 2:
            self.shape = ( self.sample_count,2)
        else:
            self.shape = ( self.sample_count,)

    def __len__(self):
        return self.sample_count

    def __array__(self):
        return self.__getitem__(slice(None,None))

    def __getitem__(self,slc):
        n = self.sample_count
        nf = self.bytes_per_frame
        f = self.f
        if type(slc) is slice:
            start,stop,stride = slc.indices(self.sample_count)
            if stride != 1:
                raise ValueError, "stride must be 1"
            recs = max(0,stop-start)
            if recs:
                f.seek( self.datapos + nf*start)
                d = f.read( recs*nf)
            else:
                d = ''
            return self.arr_decoder( d,recs)

        else:   # must be an integer
            i = int(slc)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise IndexError
            f.seek(self.datapos + nf*i)
            return self.samp_decoder(f.read(nf))

    def __repr__(self):
        return "WavFileReader(bit_per_samp=%d, nchans=%d, sample_rate=%d, sample_count=%d)"%(
            self.bits_per_samp, self.nchans, self.sample_rate, self.sample_count )

    __str__ = __repr__

######################

def dec_8_mono(s):
    return (ord(s)-128)*256
def dec_8_stereo(s):
    r = map(ord,s)
    return np.array( ((r[0]-128)*256, (r[1]-128)*256), 'int16')

def dec_8_mono_arr(s,n):
    a = np.fromstring(s,np.uint8)
    return (a.astype(np.int16)-128)*256

def dec_8_stereo_arr(s,n):
    a = np.fromstring(s,np.uint8)
    a.shape = (n,2)
    return (a.astype(np.int16)-128)*256

def dec_16_mono(s):
    return struct.unpack('<h', s )[0]
def dec_16_stereo(s):
    return np.array(struct.unpack('<2h', s ), 'int16')

if Need_byte_swap:
    def dec_16_mono_arr(s,n):
        return np.fromstring(s,np.int16).byteswapped()

    def dec_16_stereo_arr(s,n):
        a = np.fromstring(s,np.int16).byteswapped()
        a.shape = (n,2)
        return a
else:
    def dec_16_mono_arr(s,n):
        return np.fromstring(s,np.int16)

    def dec_16_stereo_arr(s,n):
        a = np.fromstring(s,np.int16)
        a.shape = (n,2)
        return a


##########################################################
"""
WaveFileWriter
 Creating the object opens the output file, and configures
 it. Configuration can be via passing in a reference object
 (WaveFileWriter or WaveFileReader), or by keyword params:
     sample_rate = ...       (def 44100)
     nchans = 1 or 2         (def 1)
     bits_per_samp = 8...16  (def 16)

 (the filename parameter can also be a file open for binary write)

 If kw parms and a reference object are present, the kw params
 override.

 After creating the object, perform writes via 'setitem',
 e.g.
   w = WaveFileWriter( 'out.wav', sample_rate=16000)
  w[0] = 100        # first sample
  w[1:4] = (200,300,400)
...
  w.close()

Currently, first write must start at locn 0, and all following
writes must immediately follow previous write.
Also, negative indices may not be used on the slice. 
it is possible to use an indefinite end index,
 e.g.
    w[0] = 100
    w[1:]= (200,300,400)   # [1:4]
    w[4:] = (500,600)      # [4:6]


 Also allowed, for appending:

   w.write_samples( ( 500, 600,-200,400) )  # add 4 samples to end 

The examples show data supplied as tuples of ints, but
any sequence type works; and numpy arrays of int16 are the
most efficient. If the file is stereo, the assignments
should supply 2-d sequences.

Regardless of the bits_per_samp, data should be supplied to
range over -32768  .. 32767; it will be scaled as needed.

This is how to make a writer which is configured the same as a reader:
    wr = WaveFileReader("in.wav")
    ww = WaveFileWriter("out.wav", wr)
    .. can now read from wr and write to ww.

"""
numpy_array_type = type(np.array([0]))
                        
class WaveFileWriter(object):

    __slots__= ['nchans', 'f',
        'bits_per_samp',
         'bytes_per_frame',
         'sample_rate',
         'datapos',     # position in file
         'curindex'
         ]

    def __init__(self, fout, refobj=None, **kw):
        if refobj is not None:
            bits_per_samp = refobj.bits_per_samp
            nchans = refobj.nchans
            sample_rate = refobj.sample_rate
        else:
            bits_per_samp = 16
            nchans = 1
            sample_rate = 44100
        if len(kw):
            kw = kw.copy()
            if kw.has_key( 'bits_per_samp'):
                bits_per_samp = kw.pop('bits_per_samp')
            if kw.has_key( 'nchans'):
                nchans = kw.pop('nchans')
            if kw.has_key( 'sample_rate'):
                sample_rate = kw.pop('sample_rate')
            if len(kw):
                raise ValueError, "unknown paramaters: " + ' '.join(kw.keys())
        self.bits_per_samp = bits_per_samp
        self.nchans = nchans
        self.sample_rate = sample_rate

        if nchans not in (1,2) or not ( 8<= bits_per_samp <= 16):
            raise ValueError, "bad nchans or bits_per_samp"
        bpf = nchans
        if bits_per_samp > 8:
            bpf *= 2
        self.bytes_per_frame = bpf
        self.curindex = 0

        if hasattr(fout,'write'):
            f= fout
        else:
            f = open(fout,'wb')
        #
        # start writing
        #
        f.write('RIFFxxxxWAVEfmt ')
        p = 16
        d = struct.pack('<LhhLLhh', 16, 1, nchans,
            sample_rate, sample_rate * bpf, bpf, bits_per_samp)
        f.write(d)
        p += len(d)
        d = 'fact\x04\0\0\0xxxxdataxxxx'
        f.write(d)
        p += len(d)
        self.datapos = p
        self.f = f
        #
        # will need to go back and modify 3 'xxxx' fields
        #

    def close(self):
        if self.f is None:
            return
        n = self.curindex
        p = self.datapos

        dlen = self.bytes_per_frame * n
        d2 = struct.pack('<L4sL', n, 'data', dlen )
        d1 = struct.pack('<L', dlen + p-8)
        self.f.seek(4)
        self.f.write(d1)
        self.f.seek(p-12)
        self.f.write(d2)
        self.f = None

    def __del__(self):
        self.close()


    def __setitem__(self, slc,val ):
        is_slc = type(slc) is slice
        if is_slc:
            if slc.step is not None and slc.step != 1:
                raise IndexError, "slice step must be 1"
            start = slc.start
            stop = slc.stop
            if start is None: start = 0
        elif type(slc) in (int, long):
            start = slc
            stop = slc+1
            val = [ val ]
        else:
            raise IndexError, "bad index"
        if start < 0 or ( stop is not None and ( stop < start )):
            raise IndexError, "invalid slice parameters"

        if stop is None:
            stop = start + len(val)
        else:
            if len(val)+start != stop:
                raise ValueError, "size mismatch in slice assignment"

        if self.curindex != start:
            raise ValueError, "WaveFileWriter assignments must be in seq"
        self.curindex = stop

        #
        # figure out if 'val' is already an array.
        #
        if not( type(val) is numpy_array_type and val.dtype == np.int16):
            val = np.asarray(val, np.int32)
            val = np.clip(val,-32768, 32767).astype(np.int16)
        #
        # test shape of array
        #
        shp = (stop-start,)
        if self.nchans == 2:
            shp = shp + (2,)
        if val.shape != shp:
            raise ValueError, "assigning array shape %s to slice of shape %s"%(
                val.shape, shp)
        #
        # convert to string
        #
        if self.bits_per_samp == 8:
            val = (((val+128)>>8)+128).astype(np.uint8)
        else:
            if Need_byte_swap:
                val = val.byteswapped()
        #
        val = val.tostring() # is a buffer now
        #
        self.f.seek(self.datapos + start * self.bytes_per_frame)
        self.f.write(val)


    def write_samples(self, seq ):
        self.__setitem__( slice( self.curindex, None), seq )

    def __repr__(self):
        return "WavFileWriter(bits_per_samp=%d, nchans=%d, sample_rate=%d, curindex=%d)"%(
            self.bits_per_samp, self.nchans, self.sample_rate, self.curindex)

    __str__ = __repr__


