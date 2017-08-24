import h5py, warnings, os, glob, sys
import numpy as np
from   struct import calcsize, unpack

def fread(f, fmt):
    """
    Shengtais method to read in a predefined
    list of binary data, where the format
    is defined as the string `fmt`, such as 'fff'.
    """
    u = unpack(fmt, f.read(calcsize(fmt)))
    if (len(fmt)==1):
        u = u[0]
    return u

class data_dict(dict):
    """
    This creates a dict-like class, where all entries can also be accessed
    as attributes. Inherits from dict class, similar to the following:

    """
    __doc__ += dict.__doc__
    def __init__(self, *args, **kwargs):
        super(data_dict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_input(fname,assignment='=',comment='#',headings='<.*?>',separators=[','],skiprows=0):
    """
    Reads input parameters from a text file.

    Arguments:
    ----------

    fname : string
    :   path/name of input file

    Keywords:
    ---------

    assignment : string
    :   which character is used as assignment. commonly '=' or ':'

    comment : string
    :   which character starts a comment - rest of line
        after this character is ignored

    headings : string
    :   define a pattern that is ignored as well.

    separators : string
    :   which characters separate the assignment.
        Typically ';' or ','. Newlines always count.

    skiprows : int
    :   how many rows to skip in the beginning

    Output:
    -------
    a dictionary with all parsed parameter/value pairs.

    Example:
    --------

    The input should always be something like

        <some heading>
        varable_name [assignment] value [separator] [comment] bla bla
        [comment] bla bla

        <other heading>
        varable_name [assignment] value [separator]
        varable_name [assignment] value [separator]

    Headings will be ignored, both newlines and separators will be used
    to separate variable assignments, so this

        var1 = 1, var2 = 2 # comment
        var3 = 3

    would be parsed to

        {'var1':1,'var2':2,'var3':3}

    """
    variables = {}
    with open(fname) as f: data = f.readlines()
    # remove comments
    fct  = lambda line: line.strip().split(comment)[0]
    data = [fct(line) for line in data if fct(line)!='']
    data = data[skiprows:]
    # remove headings
    fct = lambda line: re.subn(headings,'',line)[0].strip()
    data = [fct(line) for line in data if fct(line)!='']
    # split everything
    for separator in separators:
        data_split = []
        for line in data:
            data_split += line.split(separator)
        data = data_split
    # parse variables
    for line in data:
        varname,varval = line.split(assignment)
        varname = varname.strip()
        varval  = varval.strip()
        #
        # select format
        #
        try:
            varval = int(varval)
        except ValueError:
            try:
                varval = float(varval)
            except ValueError:
                pass
        variables[varname]=varval
    return variables


def read_data(directory='.', n=-1, igrid=0, fname=None, log_grid=0):
    #
    # construct the binary filename & file path
    #
    if n==-1:
        filenames = glob.glob(os.path.join(os.path.expanduser(directory),'bin_data','bin_out*'))
        if len(filenames) == 0:
            raise ValueError('no binary file found')
        filename_full = filenames[-1]
        filename      = os.path.split(filename_full)[-1]
    else:
        filename = 'bin_out{:04d}'.format(n)
        filename_full = os.path.join(os.path.expanduser(directory),filename)
    #
    # read initial entries that define what is to be read in next
    #
    with open(filename_full,'rb') as f:
        nx4     = fread(f, 4*"i")
        time    = fread(f, 1*"f")
        bbox    = fread(f, 5*"f")
        nplanet = int(bbox[4]);

        if (nplanet > 0):
            planet_info = fread(f,(3*nplanet)*"f");
            rp          = np.zeros(nplanet)
            phip        = np.zeros(nplanet)
            pmass       = np.zeros(nplanet)
            ii = 0;
            for i in range(0, nplanet):
                rp[i]    = planet_info[ii+0];
                phip[i]  = planet_info[ii+1];
                pmass[i] = planet_info[ii+2];
                ii += 3;

        disk_info = fread(f, 4*"f");
        cs    = disk_info[0];
        beta  = disk_info[1];
        zeta  = disk_info[2];
        Mdisk = disk_info[3];

        nvar  = nx4[0]
        nx    = nx4[1]
        ny    = nx4[2]
        nproc = nx4[3]
        dx    = (bbox[1]-bbox[0])/nx;
        dy    = (bbox[3]-bbox[2])/ny;
        na    = (nvar-4)//3
        #
        # construct the grid
        #
        if igrid == 0:
            x = bbox[0] + (np.arange(0,nx)*1.0/nx + 0.5/nx)*(bbox[1]-bbox[0]);
            if (log_grid == 1):
                logdr = (np.log(bbox[1])-np.log(bbox[0]))/nx
                logx  = np.log(bbox[0]) + (np.arange(0,nx)*1.0 + 0.5)*logdr
                x     = np.exp(logx)
            y      = bbox[2] + (np.arange(0,ny+1)*1.0/ny + 0.5/ny)*(bbox[3]-bbox[2]);
            xx, yy = np.meshgrid(x, y)
            xy1    = xx*np.cos(yy)
            xy2    = xx*np.sin(yy)
            igrid  = 1
            data   = np.zeros((nx,ny+1,nvar), dtype=np.float32, order="F")
        #
        # reading in the data
        #
        for i in range(0, nproc):
            sys.stdout.write('\rreading part {} of {}\n'.format(i+1,nproc))
            sys.stdout.flus()
            n4   = fread(f, 4*"i")
            ix   = n4[0]   # starting x-pos
            iy   = n4[1]   # starting y-pos
            nx1  = n4[2]   # number cell in x
            ny1  = n4[3]   # number cell in y
            dat1 = fread(f, nvar*nx1*ny1*"f")
            dat1 = np.array(dat1).reshape((nx1,ny1,nvar), order="F")
            data[ix:ix+nx1,iy:iy+ny1,:] = dat1.copy();
            del dat1
        print('Done reading data')
    #
    # read in parameters
    #
    input_file = os.path.join(directory,'planet2D_coag.input')
    params = read_input(input_file)
    #
    # assign the variables to fields
    #
    d  = {  'n':       n,
            'x':       x,
            'xx':      xx,
            'dx':      dx,
            'y':       y,
            'yy':      yy,
            'dy':      dy,
            'time':    time,
            'cs':      cs,
            'beta':    beta,
            'zeta':    zeta,
            'Mdisk':   Mdisk,
            'nx':      nx,
            'ny':      ny,
            'xy1':     xy1,
            'xy2':     xy2,
            'nproc':   nproc,
            'nvar':    nvar,
            'sigma_g': data[:,:,0].reshape((nx,ny+1),order='F'),
            'P_gas':   data[:,:,1].reshape((nx,ny+1),order='F'),
            'vr_g':    data[:,:,2].reshape((nx,ny+1),order='F'),
            'vp_g':    data[:,:,3].reshape((nx,ny+1),order='F'),
            'sigma_d': data[:,:,1+3*np.arange(na)].reshape((nx,ny+1,na),order='F'),
            'vr_d':    data[:,:,2+3*np.arange(na)].reshape((nx,ny+1,na),order='F'),
            'vp_d':    data[:,:,3+3*np.arange(na)].reshape((nx,ny+1,na),order='F'),
            'params':  params
         }
    #
    # if a file name was given, we store (or add) the data in a hdf5 file
    #
    if fname is not None:
        #
        #create a hdf5 file
        #
        with h5py.File(fname,'a') as f:
            #
            # check for existing data,
            # pick an unused data name
            #
            g_name = 'data_%04i'%n
            #
            # create a group if it doesn't exist
            #
            if g_name in f:
                g = f[g_name]
            else:
                g = f.create_group(g_name)
            #
            # store grain size and folder as a group attribute
            #
            if a==None:
                g.attrs['grainsize'] = np.nan
                warnings.warn('no grain size attribute was set for this data group')
            else:
                g.attrs['grainsize'] = a
            g.attrs['folder']    = directory
            #
            # store the data in our group; overwrite if it exists
            #
            for k,v in d.items():
                if k in g: del g[k]
                g.create_dataset(k,data=v)