'''
A class to read a sequence of tiff or an .mp4 and extract cell images.
'''
import imageio
from glob import glob
import os
import numpy as np
from scipy import stats
import zipfile

__all__ = ['Movie', 'MovieFolder', 'MovieFile', 'MovieZip']

_INFINITE_SIZE = 1000000000 # maximum number of frames

class Movie(object):
    '''
    Movie.

    Arguments:
        * filename: name of movie file (e.g. mp4) or folder with images (tiff or png)
        * pixel_size: size of a pixel in um
        * gray: if True, convert to gray scale by just selecting the red channel (faster)
        * step: frame increment
        * auto_invert: if True, inverts if with white background
    '''
    def __init__(self, filename=None, pixel_size=None, gray=False, step=1, auto_invert=False):
        self.pixel_size = pixel_size
        self.gray = gray
        self.start = 0
        self.end = _INFINITE_SIZE
        self.step = step
        self.n = _INFINITE_SIZE # number of frames

        self.filename = filename
        self.position = 0
        self.invert = False
        self.auto_invert = auto_invert

    def set_auto_invert(self):
        if self.auto_invert:
            image = self.current_frame()
            most_frequent_value, _ = stats.mode(image.flatten())
            if image.mean() > np.iinfo(image.dtype).max/2:  # white
                self.invert = True

    def select_file(self):
        pass

    def current_frame(self):
        frame = self._current_frame()
        if self.gray and frame.ndim == 3:
            frame = frame[:,:,0] #.mean(axis=2)
        if self.invert:
            return np.iinfo(frame.dtype).max-frame
        else:
            return frame

    def _current_frame(self):
        pass

    def next_frame(self):
        frame = self.current_frame()
        self.position+=self.step
        return frame

    def __len__(self):
        return self.n

    def frames(self, verbose=False, n=1):
        '''
        Generator that yields frames.
        If `verbose` is True, displays frame number every 10 frames.
        n = number of frames (if n>1, returns a list of frames)
        '''
        # If gray is True, then returns images in gray scale

        if n == 1 :
            while True:
                try:
                    if (self.position>=self.end):
                        return

                    yield self.next_frame()
                    if verbose and (self.position % 10 == 0):
                        print('Frame {}/{}'.format(self.position, len(self)))
                except: # IndexError, and for movie?
                    return
        else:
            while True:
                if (self.position >= self.end):
                    return
                frame_list = []
                try:
                    for _ in range(n):
                        frame_list.append(self.next_frame())
                        if verbose and (self.position % 10 == 0):
                            print('Frame {}/{}'.format(self.position, len(self)))
                except:
                    pass
                if len(frame_list)==0:
                    return
                else:
                    yield frame_list

    def close(self):
        pass

# Movie file
class MovieFile(Movie):
    '''
    Movie file.

    Arguments:
        * filename: name of folder withimages (tiff or png)
        * fps: frames per second
        * pixel_size: size of a pixel in um
    '''
    def __init__(self, filename, **kwds):
        Movie.__init__(self, filename, **kwds)

        self.reader = imageio.get_reader(filename)
        self.set_auto_invert()

    def __len__(self): # number of frames
        if Movie.__len__(self) == _INFINITE_SIZE:
            length = self.reader.get_length()
            if type(length) != type(0):
                self.n = _INFINITE_SIZE
            else:
                self.n = length
        return self.n

    # Generator giving frames one by one
    def _current_frame(self):
        return self.reader.get_data(self.position)

    def close(self):
        self.reader.close()

# Movie folder
class MovieFolder(Movie):
    '''
    Movie folder.

    Arguments:
        * filename: name of folder withimages (tiff or png)
        * fps: frames per second
        * pixel_size: size of a pixel in um
    '''
    def __init__(self, filename, **kwds):
        Movie.__init__(self, filename, **kwds)

        # Get all file names
        filename+='/'
        self.files = glob(filename+'*.tif')+glob(filename+'*.tiff')+glob(filename+'*.png') # we could add other extensions

        if len(self.files)==0:
            raise FileNotFoundError('The folder {} is empty'.format(filename))

        self.files.sort()
        self.set_auto_invert() # not the best way to do it

    def __len__(self): # number of frames
        return len(self.files)

    # Generator giving frames one by one
    def _current_frame(self):
        return imageio.imread(self.files[self.position])

# Movie folder, zipped
class MovieZip(MovieFolder):
    '''
    Movie folder.

    Arguments:
        * filename: name of zipped file withimages (tiff or png)
        * fps: frames per second
        * pixel_size: size of a pixel in um
    '''
    def __init__(self, filename, **kwds):
        Movie.__init__(self, filename, **kwds)

        self.zipname = filename
        # Get all file names
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            # List all files in the zip
            self.files = zip_ref.namelist() # assuming these are only images

        self.files.sort()
        self.set_auto_invert() # not the best way to do it

    def __len__(self): # number of frames
        return len(self.files)

    # Generator giving frames one by one
    def _current_frame(self):
        with zipfile.ZipFile(self.zipname, 'r') as zip_ref:
            with zip_ref.open(self.files[self.position]) as file:
                image = imageio.imread(file)
        return image
