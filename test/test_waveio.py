
# Basic test for wave I/O
#
import sys
sys.path[:0] = ["../src"]

from waveio import WaveFileReader, WaveFileWriter
import numpy as np


def runtests():
    run_sweep_tests()

def run_sweep_tests():
    MakeFreqSweep( "sweep_16b_mono_44k.wav", fsamp=44100, bits_per_samp=16, stereo=False)
    MakeFreqSweep( "sweep_16b_stereo_44k.wav", fsamp=44100, bits_per_samp=16, stereo=True)
    MakeFreqSweep( "sweep_8b_mono_32k.wav", fsamp=32000, bits_per_samp=8, stereo=False)
    MakeFreqSweep( "sweep_8b_stereo_32k.wav", fsamp=32000, bits_per_samp=8, stereo=True)

    expect_rms = 24000* np.sqrt(0.5)

    #----------------------------------
    wr = WaveFileReader("sweep_16b_mono_44k.wav")
    if wr.nchans != 1 or wr.sample_rate != 44100 or wr.bits_per_samp != 16 or wr.shape != (44100*12,):
        raise ValueError("failed to make sweep_16b_mono_44k.wav")
    # get the last second
    data = wr[-44100:].copy()
    if data.shape != (44100,) or data.dtype != 'int16': raise ValueError

    rms = np.sqrt((data.astype('float32')**2).mean())
    #print data.mean(), rms

    if abs(data.mean()) > 1.0:   raise ValueError("bad mean")
    if abs( rms-expect_rms) > 0.8: raise ValueError("bad rms")
    #----------------------------------
    wr = WaveFileReader("sweep_16b_stereo_44k.wav")
    if wr.nchans != 2 or wr.sample_rate != 44100 or wr.bits_per_samp != 16 or wr.shape != (44100*12,2):
        raise ValueError("failed to make sweep_16b_stereo_44k.wav")
    datam = data
    # get the last second
    data = wr[-44100:]
    if data.shape != (44100,2) or data.dtype != 'int16': raise ValueError
    # 0 channel should be identical to datam
    if (data[:,0] != datam).any(): raise ValueError( "mono/stereo mismatch")

    dataf= data.astype('float32')

    rms = np.sqrt((dataf[:,1]**2).mean())
    prod = ( dataf[:,0]*dataf[:,1]).mean()
    #print dataf[:,1].mean(), rms,prod

    if abs(dataf[:,1].mean()) > 1.0:   raise ValueError("bad mean")
    if abs( rms-expect_rms) > 0.8: raise ValueError("bad rms")
    if abs( prod+3960) > 200: raise ValueError("bad prod")

    #----------------------------------
    wr = WaveFileReader("sweep_8b_mono_32k.wav")
    if wr.nchans != 1 or wr.sample_rate != 32000 or wr.bits_per_samp != 8 or wr.shape != (32000*12,):
        raise ValueError("failed to make sweep_8b_mono_32k.wav")

    data = wr[-32000:].copy()
    if data.shape != (32000,) or data.dtype != 'int16': raise ValueError

    rms = np.sqrt((data.astype('float32')**2).mean())
    #print data.mean(), rms

    if abs(data.mean()) > 3.0:   raise ValueError("bad mean")
    if abs( rms-expect_rms) > 6: raise ValueError("bad rms")
    #----------------------------------
    wr = WaveFileReader("sweep_8b_stereo_32k.wav")
    if wr.nchans != 2 or wr.sample_rate != 32000 or wr.bits_per_samp != 8 or wr.shape != (32000*12,2):
        raise ValueError("failed to make sweep_8b_stereo_32k.wav")
    datam = data
    # get the last second
    data = wr[-32000:]
    if data.shape != (32000,2) or data.dtype != 'int16': raise ValueError
    # 0 channel should be identical to datam
    if (data[:,0] != datam).any(): raise ValueError( "mono/stereo mismatch")

    dataf= data.astype('float32')

    rms = np.sqrt((dataf[:,1]**2).mean())
    prod = ( dataf[:,0]*dataf[:,1]).mean()
    #print dataf[:,1].mean(), rms,prod

    if abs(data[:,1].mean()) > 3.0:   raise ValueError("bad mean")
    if abs( rms-expect_rms) > 6: raise ValueError("bad rms")
    if abs( prod+2224) > 200: raise ValueError("bad prod")


# make a frequency sweep from 15 Hz to 15 kHz in 12 seconds

def MakeFreqSweep( fname,
    fsamp = 44100,
    freq0 = 15.0,
    freq1 = 15000.0,
    runtime = 12.0 ,
    bits_per_samp=16,
    ampl = 24000,
    stereo = False):

    winlen = int(round(0.020*fsamp))

    # done as sin( A* (exp(B * t )-1))
    #  freq(t) = A*B*exp(B*t)/(2pi)
    # so we need to solve
    #             A*B/2pi = f0
    #             A*B*(exp(B*runtime))/2pi = f1
    #  -> exp(B*runtime) = f1/f0
    #
    B = np.log( freq1/freq0)/runtime
    A = freq0*2*np.pi/B
    nsamps = int(round(runtime*fsamp))
    nchans = 1
    if stereo:
        nchans=2    
    ww = WaveFileWriter( fname, bits_per_samp = bits_per_samp, sample_rate = fsamp, nchans =nchans)
 
    CHUNK = 65535
    ipos = 0
    # (don't even try to do this in single-precision floats...)
    while ipos < nsamps:
        nsampnow = min(CHUNK, nsamps-ipos)
        t = np.arange( ipos,ipos+nsampnow)* (B/fsamp)
        theta = np.expm1(t) * A
        if not stereo:
            y = np.sin( theta ) * float(ampl)
        else:
            # left is sin;right is cos with window
            y = np.zeros( (nsampnow,2), 'float')
            y[:,0] = np.sin(theta) * float(ampl)
            y1 = np.cos(theta) * float(ampl)
            if ipos == 0:       # apply window
                winfunc = np.sin( np.arange(winlen)*(np.pi*0.5/winlen))**2
                y1[:winlen] *= winfunc
            y[:,1] = y1
        yi = np.round( y ).astype('int16')
        ww.write_samples(yi)
        ipos += nsampnow
    ww.close()



runtests()

