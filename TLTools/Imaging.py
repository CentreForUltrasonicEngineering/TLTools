import time
import sys
import os
import inspect
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from TLTools import parula
from TLTools import gpustruct
import warnings


class PyTFM:
    def __init__(self):
        # Load up our kernel
        with open(os.path.dirname(inspect.getfile(gpustruct))+'/'+'TFMKernel.cu', 'r') as KernelFile:
            kernel_string = KernelFile.read()
        to_include = []
        try:
            if sys.getwindowsversion().major is 10:
                to_include.append('C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.10240.0\\ucrt')
        except AttributeError:
            pass
        self.Kernel = SourceModule(kernel_string, include_dirs=to_include)

        # Initialise some empty variables
        self.Array = []
        self.FMC = []
        self.V2 = []

        self.Params = gpustruct.GPUStruct([(np.float32, 'x1', 0), # Placeholder
                                    (np.float32, 'y1', 0),  # Placeholder
                                    (np.float32, 'z1', 0),  # Placeholder
                                    (np.float32, 'x2', 0),  # Placeholder
                                    (np.float32, 'y2', 0),  # Placeholder
                                    (np.float32, 'z2', 0),  # Placeholder
                                    (np.float32, 'slow1', 0),  # Placeholder
                                    (np.float32, 'slow2', 0),  # Placeholder
                                    (np.float32, 'c0', 0),  # Placeholder
                                    (np.float32, 'c1', 0),  # Placeholder
                                    (np.float32, 'c2', 0),  # Placeholder
                                    (np.float32, 'c3', 0),  # Placeholder
                                    (np.float32, 'c4', 0),  # Placeholder
                                    (np.float32, 'c5', 0),  # Placeholder
                                    (np.float32, 'c6', 0),  # Placeholder
                                    (np.float32, 'c7', 0),  # Placeholder
                                    (np.float32, 'c7', 0),  # Placeholder
                                    (np.float32, '*DataVector', np.zeros(100, dtype=np.float32)),
                                    (np.int32, 'DataVectorElementCount', 100), ])

        # Other variables
        self.blockSize = 128
        self.refractionType = 0

        # Check what we've done
        self.doneFMCupload = 0
        self.TLFMC = 0
        self.doneProbeupload = 0
        self.setparams = 0
        self.setimage = 0
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0

        # Unnecessary stuff to keep PyCharm happy...
        self.n_elem = 0
        self.sample_length = 0
        self.Fs = 0
        self.Ts = 0

    def TLuploadFMC(self, FMC):
        thisFMC = FMC.get_FMC()
        x = np.sqrt(thisFMC.shape[0])
        self.n_elem = np.floor(x).astype(np.int32)
        self.sample_length = np.int32(len(thisFMC[0]))
        self.FMC = thisFMC.astype(np.float32).flatten()
        self.Fs = np.float32(FMC.Fs)
        self.Ts = np.float32(FMC.time_start)
        self.doneFMCupload = 1
        self.TLFMC = 1
        self.donetfm=0
        self.donelog=0

    def upload_FMC(self, FMC):
        x = np.sqrt(FMC.shape[0])
        self.n_elem = np.floor(x).astype(np.int32)
        self.sample_length = np.int32(len(FMC[0]))
        if not x == np.floor(x) or len(FMC.shape) is not 2:
            raise Exception('Expected an FMC of dimensions n^2 by s, where n is the number of elements and s is the number of samples')
        self.FMC = FMC.astype(np.float32).flatten()
        self.doneFMCupload = 1
        self.donetfm=0
        self.donelog=0

    def buildProbe(self,pitch):
        if not self.doneFMCupload:
            raise Exception('Upload the FMC first so we know how big to make the array')
        start = 0
        stop = pitch*(self.n_elem-1)
        y_dim = np.linspace(start,stop,self.n_elem)
        self.Array = np.zeros((3,self.n_elem)).astype(np.float32)
        self.Array[1] = y_dim
        self.Array[1] = self.Array[1] - np.mean(self.Array[1])
        self.ArrayGPU = self.Array.T.flatten()
        self.doneProbeupload = 1
        self.setimage = 0
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0
        
    def uploadProbe(self, Array):
        if not self.doneFMCupload:
            raise Exception('Upload the FMC dataset first')
        self.Array = Array.astype(np.float32)
        if not Array.shape == (3, self.n_elem):
            raise Exception('Probe locations must be an array in the format 3 by n, where n is the number of elements')
        self.Array[1] = self.Array[1] - np.mean(self.Array[1])
        # We need the unflattened array for later
        self.ArrayGPU = self.Array.T.flatten()
        self.doneProbeupload = 1
        self.setimage = 0
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0

    def raiseArrayToHeight(self,distance):
        if not self.doneProbeupload:
            raise Exception('Upload the array first')
        self.Array[2,:] = distance
        self.ArrayGPU = self.Array.T.flatten()
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0

    def setParameters(self,**kwargs):
        try:
            self.V1 = np.float32(kwargs['Velocity1'])
            self.Params.slow1 = 1/self.V1
        except KeyError:
             raise Exception('One or more neccessary variables were not defined')

        if 'Ts' not in kwargs and self.TLFMC is 0:
            raise Exception('Time start was not defined')
        elif 'Ts' in kwargs:
            self.Ts = np.float32(kwargs['Ts'])

        if 'Fs' not in kwargs and self.TLFMC is 0:
            raise Exception('FMC Sampling frequency was not defined')
        elif 'Fs' in kwargs:
            self.Fs = np.float32(kwargs['Fs'])

        if 'Velocity2' in kwargs:
            self.V2 = kwargs['Velocity2']
            self.Params.slow2 = 1/self.V2

        self.setparams = 1
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0

    def setRefraction(self,**kwargs):
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0
        if 'Velocity2' in kwargs:
            self.V2 = kwargs['Velocity2']
            self.Params.slow2 = 1/self.V2

        if 'RefractionType' not in kwargs:
            self.refractionType = 0
            return

        testStr = str.lower(kwargs['RefractionType'])
        if testStr == 'none':
            self.refractionType = 0

        elif testStr == 'flat':
            self.refractionType = 1

        elif testStr == 'peritem':
            self.refractionType = 2
        else:
            raise Exception('Unknown refraction type')

        if self.refractionType is not 0 and not self.V2:
            raise Exception('You need to define a second wave speed before you enable refraction')

    def setImage(self, **kwargs):
        # this is to enable a few different ways the image location can be defined:
        squares = ['nPixels', 'yExtend', 'zExtend']
        anchor1 = ['y0', 'dy', 'y1', 'z0', 'dz', 'z1']
        anchor2 = ['y0', 'ny', 'y1', 'z0', 'nz', 'z1']
        if all(term in kwargs for term in squares):  # check if kwargs contains all the items in "squares"
            # version where one specifies extent of the image from the y-oriented array, and from zero
            if not self.doneProbeupload:
                raise Exception('The array needs to be uploaded before these options can be used')
            self.ny = np.int32(kwargs['nPixels'])
            self.nz = np.int32(kwargs['nPixels'])
            # create y coordinates of the image as that far to the left and to the right of the edges of the array:
            self.y = np.linspace(min(self.Array[1])-kwargs['yExtend'],max(self.Array[1])+kwargs['yExtend'],self.ny).astype(np.float32)
            # create z coordinates of the image as that far in Z direction from zero.
            self.z = np.linspace(0,kwargs['zExtend'],self.nz).astype(np.float32)

        elif all(term in kwargs for term in anchor1): # check if kwargs contains all the items in "anchor1"
            self.ny = np.int32(np.ceil((kwargs['y1'] - kwargs['y0'])/kwargs['dy']))
            self.nz = np.int32(np.ceil((kwargs['z1'] - kwargs['z0'])/kwargs['dz']))
            self.y = np.linspace(kwargs['y0'],kwargs['y1'],self.ny).astype(np.float32)
            self.z = np.linspace(kwargs['z0'],kwargs['z1'],self.nz).astype(np.float32)
        elif all(term in kwargs for term in anchor2):
            self.ny = np.int32(kwargs['ny'])
            self.nz = np.int32(kwargs['nz'])
            self.y = np.linspace(kwargs['y0'],kwargs['y1'],self.ny).astype(np.float32)
            self.z = np.linspace(kwargs['z0'],kwargs['z1'],self.nz).astype(np.float32)
        else:
            raise Exception('I don''t know how to deal with the parameters passed')
        if 'ny' not in kwargs:
            print('Defined an image that is',self.ny,'by',self.nz,'pixels.')
        self.TFM_image = np.zeros((self.ny,self.nz)).astype(np.float32).flatten()
        self.gridSize = int(np.ceil((self.nz*self.ny)/self.blockSize))
        self.Params.DataVector = np.zeros(self.ny,dtype=np.float32)
        self.Params.DataVectorElementCount = self.ny
        self.setimage=1
        self.donecoeffs = 0
        self.donetfm=0
        self.donelog=0

    def doCoeffs(self):
        if not self.setimage or not self.doneProbeupload or not self.setparams:
            raise Exception('We can''t calculate the coefficients without first defining everything necessary')

        # NOTE : there is a constant here. 33 == COEFFGEN_nTimePointsPerToFit + COEFFGEN_nTimePointsPerToCheck
        # NOTE 2: The 'check' points could as well be optimised-out later. I only needed that many points for research purposes!
        timeBuffer = np.zeros((self.ny,self.n_elem,33)).astype(np.float32).flatten()

        timeBufferSize = timeBuffer.nbytes
        self.TimeBuffer_gpu = cuda.mem_alloc(timeBufferSize)
        nTimePoints = len(timeBuffer)
        # NOTE: Again, there is the constant here: 33 == COEFFGEN_nTimePointsPerToFit + COEFFGEN_nTimePointsPerToCheck
        self.ZVector = np.linspace(min(self.z),max(self.z),33).astype(np.float32).flatten()
        self.Params.copy_to_gpu()
        ParamsInput = self.Params.get_ptr()
        gridsizeTime = int(np.ceil(nTimePoints/self.blockSize))
        GenerateTimePoints = self.Kernel.get_function("GenerateTimePoints")
        GetCoeffs =  self.Kernel.get_function("transform_tpoints_into_coeffs2GPU_kernel")
        total_coefflines = self.ny*self.n_elem
        # NOTE: Constant here : 5 = COEFF_SIZE
        CoeffArray = np.zeros((self.ny,self.n_elem,5)).astype(np.float32).flatten()
        CoeffSize = CoeffArray.nbytes
        self.Coeff_gpu = cuda.mem_alloc(CoeffSize)
        gridsizeCoeff = int(np.ceil(total_coefflines/self.blockSize))

        # Actually do some CUDA magic
        GenerateTimePoints(self.TimeBuffer_gpu,cuda.In(self.ZVector),cuda.In(self.y),cuda.In(self.ArrayGPU),np.int32(self.refractionType),self.n_elem,self.ny,np.int32(nTimePoints),ParamsInput,block=(self.blockSize,1,1), grid=(gridsizeTime,1))
        GetCoeffs(np.int32(total_coefflines),np.int32(self.n_elem),np.int32(self.ny),cuda.In(self.ZVector),self.TimeBuffer_gpu,self.Coeff_gpu,block=(self.blockSize,1,1), grid=(gridsizeCoeff,1))

        self.donecoeffs=1
        self.donetfm=0
        self.donelog=0

    def reportFitError(self,**kwargs):
        anchor = ['elem','y','z']
        if not all(term in kwargs for term in anchor):
            raise Exception('Pass the variables ''elem'', ''y'' and ''z'' to use this function.')
        if not self.donecoeffs:
            try:
                self.doCoeffs()
            except:
                raise Exception('The coefficients could not be calculated')
        TimeBuffer = np.zeros((self.ny, self.n_elem, 33)).astype(np.float32).flatten()
        CoeffArray = np.zeros((self.ny,self.n_elem,5)).astype(np.float32).flatten()
        cuda.memcpy_dtoh(TimeBuffer, self.TimeBuffer_gpu)
        cuda.memcpy_dtoh(CoeffArray, self.Coeff_gpu)
        CoeffBuf = np.reshape(CoeffArray, (self.ny, self.n_elem, 5))
        timebuf = np.reshape(TimeBuffer, (self.ny, self.n_elem, 33))
        i = kwargs['elem']  # Array element
        j = kwargs['y']  # Y-position
        k = kwargs['z']  # Z-position
        newtime = np.polyval(CoeffBuf[j, i, :], self.ZVector[k])
        print('Propagation time for array element', i, 'at the y-position of', self.y[j], 'and z-position of',
              self.ZVector[k], 'is', timebuf[j, i, k])
        print('Same propagation time from the coefficients is', newtime)
        error_time = np.abs(newtime - timebuf[j, i, k])
        error_samples = error_time * self.Fs
        print('Error is', np.abs(newtime - timebuf[j, i, k]), 'or', error_samples, 'samples')
        return np.abs(newtime - timebuf[j, i, k]),error_samples

    def doImage(self):
        if not self.donecoeffs:
            try:
                self.doCoeffs()
            except:
                raise Exception('The coefficients could not be calculated for this TFM image')
        TFM_coeff = self.Kernel.get_function("TFM_coeff")
        # t = time.time()
        TFM_coeff(cuda.Out(self.TFM_image), cuda.In(self.FMC), cuda.In(self.ArrayGPU), self.n_elem, self.Fs, cuda.In(self.z), self.nz, self.sample_length, self.Ts, self.Coeff_gpu,block=(self.blockSize,1,1), grid=(self.gridSize,1))
        # elapsed = time.time() - t
        # print('TFM took %s seconds' % float('%.3g' % elapsed))
        self.TFM_lin = self.TFM_image.reshape((self.ny,self.nz))
        self.donetfm = 1
        return self.TFM_lin

    def processImage(self):
        if not self.donetfm:
            try:
                self.doImage()
            except:
                raise Exception('The TFM image could not be generated at this time')
        
        TFM2 = abs(self.TFM_lin)
        TFM2 = TFM2 / max(TFM2.flatten())
        self.TFM_log = 20*np.log10(TFM2)
        self.donelog=1

    def get_log_TFM(self):
        if not self.donelog:
            try:
                self.processImage()
            except:
                raise Exception('Couldn''t produce a logarithmic TFM image')
        return self.TFM_log

    def printTFM(self,**kwargs):
        if 'type' not in kwargs:
            kwargs['type'] = 'log'
        if 'range' in kwargs:
            dyrange = kwargs['range']
        else:
            dyrange = 20    
        if kwargs['type'] == 'linear':
            if not self.donetfm:
                try:
                    self.processImage()
                except:
                    raise Exception('Couldn''t produce a linear TFM image')
            plt.imshow(self.TFM_lin,extent=(min(self.y),max(self.y),min(self.z),max(self.z)),cmap=parula.parula_map)
        else:
            if not self.donelog:
                try:
                    self.processImage()
                except:
                    raise Exception('Couldn''t produce a logarithmic TFM image')
            plt.imshow(self.TFM_log,extent=(min(self.y),max(self.y),min(self.z),max(self.z)),cmap=parula.parula_map)
            plt.clim(-dyrange,0)
        plt.colorbar()
        plt.title('TFM Image')
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)

    def printGPUstats(self):
        (free,total)=cuda.mem_get_info()
        print("Global memory occupancy:%.2f%% free"%(free*100/total))
        print("Gridsize is currently set to",self.gridSize,"for a 1D configuration with a blocksize of",self.blockSize)

