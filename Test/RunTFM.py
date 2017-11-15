# %matplotlib inline
from TLTools import Imaging
import scipy.io as sio

FMC_file = 'TestData.mat'
FMC = sio.loadmat(FMC_file)
FMC_dataset = FMC['FMC'].T
ProbeElementLocations = FMC['Array']
TStart = FMC['FMCTimeStart']

TFM = Imaging.PyTFM()
TFM.uploadFMC(FMC_dataset)
TFM.uploadProbe(ProbeElementLocations)
TFM.setParameters(Velocity1=5790, Velocity2=1496, Fs=72792400, Ts=TStart)
TFM.setRefraction(RefractionType='None', Velocity2=1496 )
TFM.raiseArrayToHeight(11e-3)
TFM.setImage(y0=-20e-3, ny=1024, y1=20e-3, z0=0, nz=1024, z1=-30e-3)
TFM.printTFM()

TFM.printGPUstats()
