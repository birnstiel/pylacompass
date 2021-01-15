"""Module to read and process LA-COMPASS simulation data"""
import h5py
import warnings
import os
import glob
import sys
import re
import json
import numpy as np
import astropy.constants as c
from struct import calcsize, unpack


def fread(f, fmt):
    """
    Shengtais method to read in a predefined
    list of binary data, where the format
    is defined as the string `fmt`, such as 'fff'.
    """
    u = unpack(fmt, f.read(calcsize(fmt)))
    if (len(fmt) == 1):
        u = u[0]
    return u


def get_snapshot_numbers(fname):
    """
    Returns a list of the available snapshots in the hdf5 file fname.
    """
    indices = []
    if not os.path.isfile(fname):
        print(f'{fname} is not a file!')
    else:
        with h5py.File(fname, 'r') as f:
            indices = [int(s.split('_')[-1]) for s in list(f)]
    return indices


class data_dict(dict):
    """
    This creates a dict-like class, where all entries can also be accessed
    as attributes. Inherits from dict class.
    """

    __doc__ += dict.__doc__

    def __init__(self, *args, **kwargs):
        """Initializes a data_dict from a dictionary"""
        super(data_dict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def get_dict(self):
        """
        Returns a copy of the dictionary containing the data.
        """
        import copy
        return copy.copy(self.__dict__)


def convert_to_cgs(dd):
    """
    Takes a dictionary in code units and returns a new dictionary
    in CGS units.

    It uses the same contents, some conversions are however not clear
    and will need to be tested, such as

    - zeta
    - beta
    - Mdisk
    - P_gas

    A couple of extra values are added for convenience, such as:

    - code-to-cgs unit conversion: m_unit, r_unit, t_unit
    - v_k: keplerian velocity
    - m_star, m_planet, m_disk

    Arguments
    ---------

    dd : data_dict
        data_dict in code units

    Output
    ------

    data_dict : another data_dict in CGS units

    Note
    ----

    This was not yet tested with a data_dict in lowmem mode. It might break, and
    could be fixed with adding [()] after each call to a dataset instead of a
    numpy array.
    """
    m_unit = c.M_sun.cgs.value
    m_star = dd.params['m_star'] * m_unit
    m_disk = dd.params['M_DISK'] * m_unit
    m_planet = dd.params['mPlanet'] * m_unit

    r_unit = c.au.cgs.value * dd.params['r0_length']
    t_unit = 1 / np.sqrt(c.G.cgs.value * m_star / r_unit**3)

    v_k = np.sqrt(c.G.cgs.value * m_star / (dd.x * r_unit))

    d = {'x': dd.x * r_unit,
         'xx': dd.xx * r_unit,
         'dx': dd.dx * r_unit,
         'time': dd.time * t_unit,
         'cs': dd.cs * r_unit / t_unit,
         'Mdisk': dd.Mdisk * m_unit,  # not sure if converted correctly
         'xy1': dd.xy1 * r_unit,
         'xy2': dd.xy2 * r_unit,
         'sigma_g': dd.sigma_g * m_disk / r_unit**2,
         'P_gas': dd.P_gas * m_unit / (r_unit * t_unit),
         'vr_g': dd.vr_g * r_unit / t_unit,
         'vp_g': dd.vp_g * r_unit / t_unit,
         'sigma_d': dd.sigma_d * m_disk / r_unit**2,
         'vr_d': dd.vr_d * r_unit / t_unit,
         'vp_d': dd.vp_d * r_unit / t_unit,
         'm_unit': m_unit,
         'r_unit': r_unit,
         't_unit': t_unit,
         'm_star': m_star,
         'm_planet': m_planet,
         'm_disk': m_disk,
         'v_k': v_k,
         'rp': dd.rp * r_unit,
         'phip': dd.phip,
         'pmass': dd.pmass * m_unit,
         }

    # to work more general, copy all values over that we haven't assigned yet
    for k, v in dd.items():
        if k not in d:
            d[k] = v

    return data_dict(d)


def read_input(fname, assignment='=', comment='#', headings='<.*?>',
               separators=[';'], array_separator=',', skiprows=0):
    """
    Reads input parameters from a text file.

    Arguments:
    ----------

    fname : string
        path/name of input file

    Keywords:
    ---------

    assignment : string
        which character is used as assignment. commonly '=' or ':'

    comment : string
        which character starts a comment - rest of line
        after this character is ignored

    headings : string
        define a pattern that is ignored as well.

    separators : string
        which characters separate the assignment.
        Typically ';' or ','. Newlines always count.

    array_separator : string
        if one variable is an array, separate values by this character

    skiprows : int
        how many rows to skip in the beginning

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
    with open(fname) as f:
        data = f.readlines()

    # remove comments

    def fct(line):
        return line.strip().split(comment)[0]
    data = [fct(line) for line in data if fct(line) != '']
    data = data[skiprows:]

    # remove headings

    def fct(line):
        return re.subn(headings, '', line)[0].strip()
    data = [fct(line) for line in data if fct(line) != '']

    # split everything

    for separator in separators:
        data_split = []
        for line in data:
            data_split += line.split(separator)
        data = data_split

    # parse variables

    for line in data:
        varname, varval = line.split(assignment)
        varname = varname.strip()
        varval = varval.strip()

        # select format

        try:
            varval = int(varval)
        except ValueError:
            try:
                varval = float(varval)
            except ValueError:
                try:
                    varval = np.fromstring(varval, sep=array_separator)
                except ValueError:
                    print('could not parse {}'.format(varname))
        variables[varname] = varval
    return variables


def read_data(directory='.', inputfile='planet2D_coag.input', n=-1, igrid=0, fname=None, log_grid=0, a=None, gridfile=None):
    """
    Function to read data of the multi-species dust+gas hydro code.

    Arguments:

    directory : string
        path to read from. Should be the simulation folder containing input and
        output files as well as the binary data folder `bin_data/`.

    inputfile : string
        filename of the parameter file to read from

    n : integer
        index of the binary snapshot to read. positive values need
        to directly specify the snapshot number like n=5->bin_data0005
        negative values:
        -1 means last available
        -2 means second to last available and so on.

    igrid : integer
        grid parameter; only igrid=0 is currently implemented

    fname : None | str
        None (default): makes nothing;
        str: specify hdf5 file name to write into

    log_grid : int
        0 : not a log grid
        1 : log grid

    a : array-like
        the particle size array if known

    gridfile : str
        if given: read radial grid from this file

    Output:
    -------
    d : data_dict
        `data_dict` object containing the relevant data fields

    if `fname` is given, also a hdf5 file is written out.

    """
    #
    # construct the binary filename & file path
    #
    if n < 0:
        filenames = sorted(glob.glob(os.path.join(os.path.expanduser(directory), 'bin_data', 'bin_out*')))
        n = n % len(filenames)
        if len(filenames) == 0:
            raise ValueError('no binary file found in {}'.format(os.path.join(os.path.expanduser(directory), 'bin_data')))
        filename_full = filenames[-1]
        filename = os.path.split(filename_full)[-1]
        n = int(filename.split('bin_out')[-1])
    else:
        filename = 'bin_out{:04d}'.format(n)
        filename_full = os.path.join(os.path.expanduser(directory), 'bin_data', filename)
        if not os.path.isfile(filename_full):
            raise NameError('no binary found in {}'.format(filename_full.replace(filename, '')))
    #
    # read initial entries that define what is to be read in next
    #
    with open(filename_full, 'rb') as f:
        nx4 = fread(f, 4 * "i")
        time = fread(f, 1 * "f")
        bbox = fread(f, 5 * "f")
        nplanet = int(bbox[4])

        if (nplanet > 0):
            planet_info = fread(f, (3 * nplanet) * "f")
            rp = np.zeros(nplanet)
            phip = np.zeros(nplanet)
            pmass = np.zeros(nplanet)
            ii = 0
            for i in range(0, nplanet):
                rp[i] = planet_info[ii + 0]
                phip[i] = planet_info[ii + 1]
                pmass[i] = planet_info[ii + 2]
                ii += 3

        disk_info = fread(f, 4 * "f")
        cs = disk_info[0]
        beta = disk_info[1]
        zeta = disk_info[2]
        Mdisk = disk_info[3]

        nvar = nx4[0]
        nx = nx4[1]
        ny = nx4[2]
        nproc = nx4[3]
        dx = (bbox[1] - bbox[0]) / nx
        dy = (bbox[3] - bbox[2]) / ny
        na = (nvar - 4) // 3
        #
        # construct the grid
        #
        if gridfile is None:
            if igrid == 0:
                x = bbox[0] + (np.arange(0, nx) * 1.0 / nx + 0.5 / nx) * (bbox[1] - bbox[0])
                if (log_grid == 1):
                    logdr = (np.log(bbox[1]) - np.log(bbox[0])) / nx
                    logx = np.log(bbox[0]) + (np.arange(0, nx) * 1.0 + 0.5) * logdr
                    x = np.exp(logx)
                y = bbox[2] + (np.arange(0, ny + 1) * 1.0 / ny + 0.5 / ny) * (bbox[3] - bbox[2])
                xx, yy = np.meshgrid(x, y)
                xy1 = xx * np.cos(yy)
                xy2 = xx * np.sin(yy)
                igrid = 1
        else:
            # if (log_grid >= 1):
            x = np.zeros(nx)
            ff = open(gridfile, 'r')
            for ir in range(nx):
                x[ir] = ff.readline()
            ff.close()

            y = bbox[2] + (np.arange(0, ny + 1) * 1.0 / ny + 0.5 / ny) * (bbox[3] - bbox[2])
            xx, yy = np.meshgrid(x, y)
            xy1 = xx * np.cos(yy)
            xy2 = xx * np.sin(yy)

        data = np.zeros((nx, ny + 1, nvar), dtype=np.float32, order="F")
        #
        # reading in the data
        #
        for i in range(0, nproc):
            sys.stdout.write('\rreading part {} of {}'.format(i + 1, nproc))
            sys.stdout.flush()
            n4 = fread(f, 4 * "i")
            ix = n4[0]   # starting x-pos
            iy = n4[1]   # starting y-pos
            nx1 = n4[2]   # number cell in x
            ny1 = n4[3]   # number cell in y
            dat1 = fread(f, nvar * nx1 * ny1 * "f")
            dat1 = np.array(dat1).reshape((nx1, ny1, nvar), order="F")
            data[ix:ix + nx1, iy:iy + ny1, :] = dat1.copy()
            del dat1
        print('\rFinished reading data.    ')
        data[:, ny, :] = data[:, 0, :]
    #
    # read in parameters
    #
    input_file = os.path.join(directory, inputfile)
    if os.path.isfile(input_file):
        params = read_input(input_file)
    else:
        params = {}

    # since dict cannot be stored in hdf5, we need to encode it as json
    # since numpy cannot be stored in json, we need to convert the arrays
    # to lists

    json_encoded_params = {}
    for k, v in params.items():
        if type(v) is np.ndarray:
            json_encoded_params[k] = v.tolist()
        else:
            json_encoded_params[k] = v
    json_encoded_params = json.dumps(json_encoded_params)

    if a is None:
        a = np.logspace(np.log10(params['size_of_dust']), np.log10(params['size_of_dust_mx']), na)

    #
    # assign the variables to fields
    #
    d = {'n': n,
         'na': na,
         'x': x,
         'xx': xx,
         'dx': dx,
         'y': y,
         'yy': yy,
         'dy': dy,
         'time': time,
         'cs': cs,
         'beta': beta,
         'zeta': zeta,
         'Mdisk': Mdisk,
         'nx': nx,
         'ny': ny,
         'xy1': xy1,
         'xy2': xy2,
         'nproc': nproc,
         'nvar': nvar,
         'sigma_g': data[:, :, 0].reshape((nx, ny + 1), order='F'),
         'P_gas': data[:, :, 1].reshape((nx, ny + 1), order='F'),
         'vr_g': data[:, :, 2].reshape((nx, ny + 1), order='F'),
         'vp_g': data[:, :, 3].reshape((nx, ny + 1), order='F'),
         # 'sigma_d': data[:, :, 1 + 3 * np.arange(na)].reshape((nx, ny + 1, na), order='F'),
         'sigma_d': data[:, :, 4 + 3 * np.arange(na)].reshape((nx, ny + 1, na), order='F'),
         'vr_d': data[:, :, 2 + 3 * np.arange(na)].reshape((nx, ny + 1, na), order='F'),
         'vp_d': data[:, :, 3 + 3 * np.arange(na)].reshape((nx, ny + 1, na), order='F'),
         'params': params,
         'json_encoded_params': json_encoded_params,
         'a': a,
         'rp': rp,
         'phip': phip,
         'pmass': pmass,
         }
    #
    # if a file name was given, we store (or add) the data in a hdf5 file
    #
    if fname is not None:
        #
        # create a hdf5 file
        #
        with h5py.File(fname, 'a') as f:
            #
            # check for existing data,
            # pick an unused data name
            #
            g_name = 'data_{:04d}'.format(n)
            #
            # create a group if it doesn't exist
            #
            if g_name in f:
                g = f[g_name]
                print(f'overwriting {g_name} in {fname}')
            else:
                g = f.create_group(g_name)
                print(f'adding {g_name} to {fname}')
            #
            # store grain size and folder as a group attribute
            #
            if a is None:
                g.attrs['grainsize'] = np.nan
                warnings.warn('no grain size attribute was set for this data group')
            else:
                g.attrs['grainsize'] = a
            g.attrs['folder'] = directory
            #
            # store the data in our group; overwrite if it exists
            #
            for k, v in d.items():
                if type(v) is dict:
                    continue
                if k in g:
                    del g[k]
                g.create_dataset(k, data=v)
    #
    # end of read_data: return data dictionary
    #
    return data_dict(d)


def read_hdf5_file(fname, n=-1, lowmem=True):
    """
    To read LA-COMPASS data from a previously stored hdf5 file.

    fname : str
        file name to read in

    Keywords:
    ---------

    n : int
        select a snapshot, default is -1 which is the last snapshot

    lowmem : bool
        True:   keep the file open and use links to the datasets
        False:  read in all the data into memory and close the file

    Output:
    -------
    d, f

    d : data_dict
        data dictionary containing the data as attributes and fields

    f : h5py file handle
        the file handle to close the file manually if lowmem==True

    """
    try:
        f = h5py.File(fname, 'r')
        #
        # get the keys and transform them into integer indices
        #
        k = list(f.keys())
        indices = [int(_k.split('_')[-1]) for _k in k]
        #
        # if n is negative we count from the largest index backwards
        #
        if n < 0:
            n = sorted(indices)[n]

        # get the key corresponding to the right integer index

        key = list(f.keys())[indices.index(n)]
        print('Reading index {}'.format(n))

        # access the corresponding group and create a dictionary
        # that contains all those data

        g = f[key]
        d = {}
        for k, v in g.items():
            if lowmem:
                d[k] = v
            else:
                d[k] = v[()]

        # store the snapshot

        d['n_snapshot'] = n

        # transform the encoded parameter dictionary back from json

        d['params'] = json.loads(g['json_encoded_params'][()])
        for k, v in d.items():
            if type(v) is list:
                d[k] = np.array(v)

    finally:
        if not lowmem:
            f.close()

    # transform to data_dict and return

    return data_dict(d), f


def read_torqfile(d, torqfile):
    """
    read the data from a LA-COMPASS torq1d.dat file.

    Arguments
    ---------

    d : data_dict
        read in data from LA-COMPASS. Needed to access parameters and grid dimensions

    torqfile : str
        file path of the torq1d file

    Output
    ------
    data_dict : contains the converted quantities and the original (unconverted) data
    """

    # read file contents as lines

    with open(torqfile) as fid:
        lines = fid.readlines()

    # get the lines starting with a pound sign. there is one per snapshot
    # element 4 is the time in code units

    time_lines = [line for line in lines if line.startswith('#')]
    nt = len(time_lines)
    time = np.array([float(line.split()[4]) for line in time_lines])

    # get all the other lines which are data lines

    data = [line for line in lines if not (line.startswith('#') or line == '\n')]

    # join them to a string, then convert them to a 1D array of floats, then reshape

    data = np.fromstring(''.join(data), sep=' ').reshape(nt, d.nx, -1)

    # the entires we know, we will transform into separate arrays, but the full dataset remains in the array 'data'

    r = data[0, :, 0]
    sigmag = data[:, :, 2]
    alpha = data[:, :, 7]
    sigmad = data[:, :, 8:8 + d.na]

    # convert to cgs units

    r *= d.params['r0_length'] * c.au.cgs.value
    time *= d.params['r0_length']**1.5 / (2 * np.pi)
    factor_sig = (c.M_sun / (d.params['r0_length'] * c.au)**2).cgs.value

    sigmag *= factor_sig
    sigmad *= factor_sig
    alpha /= 1.5

    # return it as a data_dict

    return data_dict({
        'r': r,
        'orbits': time,
        'sigma_g': sigmag,
        'sigma_d': sigmad,
        'alpha': alpha,
        'data': data,
    })
