import sys
import numpy as np

# This file is designed to implement (and expand) the voxel spread function approach to B0 inhomogeneity 
# handling for GRE and ASE data, as introduced by Yablonskiy, et al. (2012)
# There are some deviations compared to the implementation in Yablonskiy. e.g. for efficiency the 
# summation over q and m are reversed in our approach.
# In the code, variables k and q also include a factor 2pi that they do not in the paper.
# To simulate long EPI readout, we have also added an extended VSF approach.

########################
# Settings and Utility #
########################

COMPLEX_TYPE = np.complex128 # Can use np.complex64 (single precision) to save memory (but somehow not computation time?)


def calculate_central_differences_in_image_axes(image_arr, voxel_size_arr=None):
    """Calculates the central differences of the image along every axis. 
    If voxel_size_arr is specified, the gradient is scaled by that distance in each axis, otherwise a spacing of 1 between voxels is assumed.

    Args:
        image_arr (ndarray): numeric, of any shape. contains image information (e.g. B0 field)
        voxel_size_arr (ndarray, optional): numeric, should be of shape (n,) where n is the number of dimensions of image_arr. 
        Defaults to None, which corresponds to a distance of 1 between voxels.

    Returns:
        tuple with n elements, where n is the number of dimensions of image_arr: Every element is an array of the same shape as
        image_arr, corresponding to the gradient of the image along the respective axis
    """
    if voxel_size_arr is None:
        voxel_size_arr = np.ones(image_arr.ndim)
    diff_arr_list = np.gradient(image_arr, *voxel_size_arr)

    return tuple(diff_arr_list)


def calculate_k(gamma_G, k_0, t):
    """Calculate k-space shift due to magnetic field gradient G at time t, given an initial displacement k_0.
    Output k will be in rad, thus include the factor 2pi, unlike the definition in Yablonkiy.
    This definition is consistent with our 2025 paper.
    Using the indicated units is recommended, but alternative unit scaling can be used at your own peril.

    Args:
        gamma_G (scalar or ndarray): The gradient magnetic field G scaled with gyromagnetic ratio to be in unit [rad Hz/mm].
        k_0 (scalar or ndarray): K-space displacement at time 0 in unit [rad/mm].
        Usually should be close to 0, unless x/y/z shimming is performed.
        t (scalar or ndarray): Time since spins were refocused (either excitation or spin echo) in unit [s].

    Returns:
        scalar or ndarray: k-space shift in unit [rad/mm]. Same shape as inputs.
    """
    return gamma_G * t + k_0


def calculate_k_grid(n_voxels, voxel_size, n_voxels_acq=None):
    """Define k-space grid in 1 dimension in unit [rad/mm].
    Grid spacing same as in Yablonskiy equation [5], but factor 2pi is integrated in k.
    Using the indicated units is recommended, but alternative unit scaling can be used at your own peril.

    Args:
        n_voxels (integer): number of voxels in k-space
        voxel_size (float): size of voxels in k-space in unit [mm]
        n_voxels_acq (integer, optional): The size of the acquisition matrix along the chosen axis. May differ from the 
        image matrix due to interpolation. Should ignore parallel imaging or partial fourier at this point.
        If the given axis is a through-plane axis in 2D imaging, choose the value 1 here, as the k-space was only sampled 
        for k=0 along that direction. Defaults to None, which means n_voxels_acq=n_voxels

    Returns:
        ndarray: k-space grid in unit [rad/mm], of shape (n_voxels,)
    """
    # Like k defined in Yablonskiy, but multiplied by 2 pi
    k_grid_arr = calculate_q_grid(n_voxels, n_voxels_acq=n_voxels_acq) * voxel_size
    return k_grid_arr


def calculate_q(gamma_G, k_0, t, voxel_size):
    """Calculate q shift due to magnetic field gradient G at time t, given an initial displacement k_0.
    q is defined as a k-space vector rescaled by the voxel size, as defined in Yablonskiy.
    Output k will be in rad, thus include the factor 2pi, unlike the definition in Yablonkiy.
    Using the indicated units is recommended, but alternative unit scaling can be used at your own peril.

    Args:
        gamma_G (scalar or ndarray): The gradient magnetic field G scaled with gyromagnetic ratio to be in unit [rad Hz/mm].
        k_0 (scalar or ndarray): K-space displacement at time 0 in unit [rad/mm].
        Usually should be close to 0, unless x/y/z shimming is performed.
        t (scalar or ndarray): Time since spins were refocused (either excitation or spin echo) in unit [s].
        voxel_size (scalar or ndarray): Size of voxels in the gradient direction in unit [mm].
        Should be acquisition size, not reconstruction size, as the acquisition is what matters for dephasing.

    Returns:
        scalar or ndarray: q shift in unit [rad]. Same shape as inputs.
    """
    return calculate_k(gamma_G, k_0, t) * voxel_size


def calculate_q_grid(n_voxels, n_voxels_acq=None):
    """Define q grid in 1 dimension in unit [rad].
    Grid spacing same as in Yablonskiy equation [5], but factor 2pi is integrated in k.
    Using the indicated units is recommended, but alternative unit scaling can be used at your own peril.

    Args:
        n_voxels (integer): number of voxels in k-space
        n_voxels_acq (integer, optional): number of actually acquired voxels in k-space (in case image is interpolated). 
        If None is provided, ignore it (i.e. no interpolation)
        n_voxels_acq (integer, optional): The size of the acquisition matrix along the chosen axis. May differ from the 
        image matrix due to interpolation. Should ignore parallel imaging or partial fourier at this point.
        If the given axis is a through-plane axis in 2D imaging, choose the value 1 here, as the k-space was only sampled 
        for k=0 along that direction. Defaults to None, which means n_voxels_acq=n_voxels

    Returns:
        ndarray: q grid in unit [rad], of shape (n_voxels,) (shape is (n_voxels_acq,), if provided)
    """
    if n_voxels_acq is None: 
        q_grid_arr = 2*np.pi * (-1./2. + (np.arange(n_voxels)+1)/n_voxels)
    else:
        # The q-space should have a number of of spaces equal to the acquired data points. But the minimum and maximum
        # values should be rescaled (compared to +-pi) to account for the lower resolution than in the reconstructed image.
        q_grid_arr = calculate_q_grid(n_voxels_acq) * n_voxels_acq/n_voxels
    return q_grid_arr


def calculate_phi(gamma_deltaB, phi_0, t):
    """Calculate the phase due to magnetic field inhomogeneity delta B0 at a time t, like in Yablonskiy equation [12].
    Using the indicated units is recommended, but alternative unit scaling can be used at your own peril.

    Args:
        gamma_deltaB (scalar or ndarray): The magnetic field off-resonance delta B0 scaled with gyromagnetic ratio to be in unit [rad Hz].
        phi_0 (scalar or ndarray): Phase at time 0 in unit [rad].
        t (scalar or ndarray): Time since spins were refocused (either excitation or spin echo) in unit [s].

    Returns:
        scalar or ndarray: phase phi in unit [rad]. Same shape as inputs.
    """
    return gamma_deltaB * t + phi_0


def shift_arr_allaxes(arr, delta_m_list):
    """Shift the values in arr by delta_m_list[i] indices along each axis index i.

    Args:
        arr (ndarray): array of any shape
        delta_m_list (list of integers): list of length arr.ndim. For each value in this list, shift the 
        arr along the corresponding axis by the provided delta_m entry.

    Returns:
        ndarray: array of same shape as arr, but with shifted values.
    """
    minus_delta_m_tuple = tuple([-delta_m for delta_m in delta_m_list])
    axis_tuple = tuple([i for i in range(len(delta_m_list))])
    shifted_arr = np.roll(arr, shift=minus_delta_m_tuple, axis=axis_tuple)
    for axis, delta_m in enumerate(delta_m_list):
        # Have to set the wrapped values to 0.
        if delta_m < 0: # If roll is towards positive indices, set beginning to 0.
            sl = [slice(None)] * shifted_arr.ndim
            sl[axis] = slice(-delta_m)
            shifted_arr[tuple(sl)] = 0.
        elif delta_m > 0: # If roll is towards negative indices, set end to 0.
            sl = [slice(None)] * shifted_arr.ndim
            sl[axis] = slice(-delta_m, None)
            shifted_arr[tuple(sl)] = 0
    return shifted_arr


def shift_arr(arr, axis, delta_m):
    """Shift the values in arr by delta_m indices along the chosen axis.
    The output has the same shape as arr.
    Values that wrap around are set to 0.
    This is the new implementation, which calls the multidim shift along one axis.

    Args:
        arr (ndarray): array of any shape
        axis (integer): index of axis along which the shift happens. Must be compatible with arr.
        delta_m (integer): integer number of voxels to define the distance of the shift.

    Returns:
        ndarray: array of same shape as arr, but with shifted values.
    """
    delta_m_list = [0 for i in range(arr.ndim)]
    delta_m_list[axis] = delta_m
    shifted_arr = shift_arr_allaxes(arr, delta_m_list)
    return shifted_arr


def apply_kspace_mask(arr, mask):
    """Transforms arr into k-space, then multiplicatively applies the mask and transforms the 
    output back into image pspace.

    Args:
        arr (ndarray): array of any shape, usually representing an n-dimensional image.
        mask (ndarray): array of same shape as arr, representing the weighting of different points in k-space.

    Returns:
        ndarray: array of same shape as arr, but after application of mask as a k-space filter.
    """
    kspace_arr = np.fft.fftshift(np.fft.fftn(arr))
    masked_kspace_arr = kspace_arr * mask
    modified_arr = np.fft.ifftn(np.fft.fftshift(masked_kspace_arr))
    return modified_arr


################################
# Functions voxel spread terms #
################################


def calculate_original_voxel_spread_function_1axis(q_background_arr, axis, delta_m_arr, n_voxels_acq=None):
    """ Calculates the voxel spread function (eta) along a single dimension, as defined in Equation [18] from Yablonskiy.
    The dependence on voxel location n is encoded in the array shape, which is taken over from q_background_arr.
    The dependence on voxel location m is encoded as an offset delta_m relative to n along the specified axis.
    This re-definition of m seemed to make the implementation more efficient, as we can apply functions to all
    different voxels n at the same time.
    In contrast to Yablonskiy's definition, we divide by the number of voxels to stay consistent with their equation [14].
    Also note that q needs to be divided by 2pi in the sinc term, since we have a different scaling of q compared to 
    Yablonskiy.
    Note that compared to Yablonskiy's definition, we flip the order of summation over q and delta_m. That is why all 
    delta_m in the given axis need to be provided at the same time in delta_m_arr. With this strategy, the structure 
    of equation is a bit different than in Yablonskiy, but this function is actually much faster (about factor 2), 
    because we can simply shift a lot of the results of the sinc-calcations rather than re-calculating them for each 
    delta_m.
    Careful: all delta_m in delta_m_arr are applied along the same axis here! This is not the same term as the 
    delta_m_list used in other functions, which contains one delta_m per axis.

    Args:
        q_background_arr (ndarray): q shift due to field gradients in unit [rad]. Should have same shape as the
        image and defines the q shift along the specified axis at each location.
        axis (int): Axis along which the voxel spread function is calculated and along which the voxel offsets 
        delta_m are applied.
        delta_m_arr (ndarray): 1 dimensional array, definining all delta_m shifts that should be evaluated along
        the given axis.
        n_voxels_acq (integer, optional): The size of the acquisition matrix along the chosen axis. May differ from the 
        image matrix due to interpolation. Should ignore parallel imaging or partial fourier at this point.
        If the given axis is a through-plane axis in 2D imaging, choose the value 1 here, as the k-space was only sampled 
        for k=0 along that direction. Defaults to None, which means n_voxels_acq=q_background_arr.shape[axis]

    Returns:
        ndarray of shape ((delta_m_arr.size) + q_background_arr.shape): Voxel spread function calculated for each 
        delta_m (first axis) and each image location (remaining axes) for the defined axis and delta_m values.
    """
    result_shape = (len(delta_m_arr),) + q_background_arr.shape
    n_voxels = q_background_arr.shape[axis]
    q_grid_arr = calculate_q_grid(n_voxels, n_voxels_acq=n_voxels_acq) # Define q values corresponding to this k-space

    result_arr = np.zeros(result_shape, dtype=COMPLEX_TYPE)
    # Equation [18] from Yablonskiy, but divide by n_voxels
    for i, q in enumerate(q_grid_arr):
        
        sinc_term = np.sinc((q-q_background_arr)/(2*np.pi))
        for i, delta_m in enumerate(delta_m_arr):
            sinc_term_m = shift_arr(sinc_term, axis, delta_m)
            result_arr[i] += sinc_term_m * np.exp(-1.0j * q * delta_m, dtype=COMPLEX_TYPE)
    result_arr /= n_voxels
    
    return result_arr


def calculate_extended_voxel_spread_function_1axis(q_background_arr, axis, delta_m_arr, n_voxels_acq=None,
                                                   q_velocity=None, g_background_arr=None, 
                                                   b0_background_arr=None, r2s_arr=None, r2p_arr=None):
    """Calculates the extended voxel spread function (eta) along a single dimension.

    This implementation extends the original voxel spread function of Yablonskiy by
    accounting for signal evolution during k-space traversal. The model incorporates:
    - Finite k-space traversal velocity (q_velocity), causing additional phase accrual
      and relaxation effects during readout.
    - Background gradient fields (g_background_arr).
    - B0 off-resonance effects (b0_background_arr).
    - R2/R2* decay (r2s_arr).
    - R2' dephasing and rephasing effects (r2p_arr).

    As in calculate_original_voxel_spread_function_1axis(), the dependence on voxel
    location n is encoded in the image dimensions and the dependence on voxel location
    m is encoded through offsets delta_m relative to n along the specified axis.
    The implementation evaluates all delta_m values simultaneously by exchanging the
    summation order over q and delta_m, improving computational efficiency.

    The formulation reduces to the original voxel spread function in the limit of
    infinite q_velocity and zero background field or relaxation effects.

    Careful: all delta_m in delta_m_arr are applied along the same axis here!
    This is not the same term as the delta_m_list used in other functions, which
    contains one delta_m array per axis.

    Args:
        q_background_arr (ndarray): q shift due to field gradients in unit [rad].
            Should have the same shape as the image and defines the q shift along
            the specified axis at each location.
        axis (int): Axis along which the voxel spread function is calculated and
            along which the voxel offsets delta_m are applied.
        delta_m_arr (ndarray): One-dimensional array defining all delta_m shifts
            that should be evaluated along the given axis.
        n_voxels_acq (integer, optional): Size of the acquisition matrix along the
            chosen axis. May differ from the image matrix due to interpolation.
            For a through-plane axis in 2D imaging, use 1.
            Defaults to None, corresponding to q_background_arr.shape[axis].
        q_velocity (float, optional): Velocity at which q-space is traversed during readout. 
            Unit should usually be rad/s.
            Defaults to np.inf, corresponding to instantaneous acquisition and therefore the original VSF formulation.
        g_background_arr (ndarray, optional): Background magnetic field gradient
            along the specified axis.
        b0_background_arr (ndarray, optional): B0 offset map in unit compatible with q_velocity. Should be scaled by gamma,
            so usually unit rad*Hz.
        r2s_arr (ndarray, optional): R2 (for ASE) or R2* (for GRE) relaxation rate map.
        r2p_arr (ndarray, optional): R2' relaxation rate map (for ASE, should be 0 for GRE).

    Returns:
        ndarray of shape ((delta_m_arr.size) + q_background_arr.shape):
            Extended voxel spread function calculated for each delta_m (first axis)
            and each image location (remaining axes).
    """
    if q_velocity is None:
        q_velocity = np.inf
        # If q_velocity not supplied, assume it is infinite (i.e. original VSF model as q/q_velocity becomes 0)
    if g_background_arr is None:
        # If g_background_arr not supplied, assume no background gradient magnetic fields
        g_background_arr = np.zeros_like(q_background_arr)
    if b0_background_arr is None:
        # If b0_background_arr not supplied, assume no b0 inhomogeneity
        b0_background_arr = np.zeros_like(q_background_arr)
    if r2s_arr is None:
        # If r2s_arr not supplied, assume no R2* (or R2) decay during readout
        r2s_arr = np.zeros_like(q_background_arr)
    if r2p_arr is None:
        # If r2p_arr not supplied, assume no R2' decay during readout
        r2p_arr = np.zeros_like(q_background_arr)

    result_shape = (len(delta_m_arr),) + q_background_arr.shape
    n_voxels = q_background_arr.shape[axis]
    q_grid_arr = calculate_q_grid(n_voxels, n_voxels_acq=n_voxels_acq) # Define q values corresponding to this k-space
    voxel_size = 3.5 # voxel size in mm. TODO: Make this properly user-selectable.

    result_arr = np.zeros(result_shape, dtype=COMPLEX_TYPE)

    # Equation [18] from Yablonskiy, but divide by n_voxels
    for i, q in enumerate(q_grid_arr):        
        # Modified sinc term to account for finite q_velocity, causing change in phase and r2* during readout
        sinc_term = np.sinc((q*(1 - g_background_arr/q_velocity*voxel_size)-q_background_arr)/(2*np.pi)).astype(COMPLEX_TYPE)
        # In case your images are GRE, set r2s to your R2* values and leave r2p (representing R2') zero.
        # In case your images are ASE, set r2s to your R2 values and r2p to your R2' values.
        # This formula should work in either case, as it models the monotonous R2/R2* decay after excitation, 
        # but dephasing and rephasing of R2' arond spin echo.
        sinc_term *= np.exp(q/q_velocity * (-r2s_arr - r2p_arr*np.sign(q/q_velocity) + 1.0j * b0_background_arr)) 
        for i, delta_m in enumerate(delta_m_arr):
            sinc_term_m = shift_arr(sinc_term, axis, delta_m)
            result_arr[i] += sinc_term_m * np.exp(-1.0j * q * delta_m, dtype=COMPLEX_TYPE)
    result_arr /= n_voxels
    
    return result_arr


def calculate_voxel_spread_function_throughplane(q_background_arr, axis, delta_m_arr):
    """Calculates the through-plane approximation of the voxel spread function.

    For axes along which only a single k-space sample was acquired
    (n_voxels_acq = 1), the voxel spread function simplifies to a shifted sinc
    function and does not require summation over q values. As in the other VSF
    implementations, all delta_m values along the specified axis are evaluated
    simultaneously for improved efficiency.

    Careful: all delta_m in delta_m_arr are applied along the same axis here!
    This is not the same term as the delta_m_list used in other functions, which
    contains one delta_m array per axis.

    Args:
        q_background_arr (ndarray): q shift due to field gradients in unit [rad]. Should have same shape as the
        image and defines the q shift along the specified axis at each location.
        axis (int): Axis along which the voxel spread function is calculated and along which the voxel offsets 
        delta_m are applied.
        delta_m_arr (ndarray): 1 dimensional array, definining all delta_m shifts that should be evaluated along
        the given axis.

    Returns:
        ndarray of shape ((delta_m_arr.size) + q_background_arr.shape): Voxel spread function for through-plane direction 
        calculated for each delta_m (first axis) and each image location (remaining axes) for the defined axis and delta_m 
        values.
    """
    result_shape = tuple([delta_m_arr.size] + list(q_background_arr.shape))
    result_arr = np.zeros(result_shape, dtype=COMPLEX_TYPE)

    sinc_term = np.sinc(q_background_arr / (2 * np.pi))
    for i, delta_m in enumerate(delta_m_arr):
        sinc_term_m = shift_arr(sinc_term, axis, delta_m)
        result_arr[i] = sinc_term_m
    return result_arr


def calculate_voxel_spread_function_1axis(q_background_arr, axis, delta_m_arr, n_voxels_acq=None,
                                          q_velocity=None, g_background_arr=None, 
                                          b0_background_arr=None, r2s_arr=None, r2p_arr=None):
    """Calculates the voxel spread function along a single axis.

    This wrapper automatically selects the appropriate voxel spread function model:

    - Through-plane approximation, if n_voxels_acq == 1.
    - Original Yablonskiy voxel spread function, if q_velocity is not supplied.
    - Extended voxel spread function, if q_velocity is supplied.

    Args:
        q_background_arr (ndarray): q shift due to field gradients in unit [rad].
            Should have the same shape as the image and defines the q shift along
            the specified axis at each location.
        axis (int): Axis along which the voxel spread function is calculated and
            along which the voxel offsets delta_m are applied.
        delta_m_arr (ndarray): One-dimensional array defining all delta_m shifts
            that should be evaluated along the given axis.
        n_voxels_acq (integer, optional): Size of the acquisition matrix along the
            chosen axis. If equal to 1, the through-plane approximation is used.
        q_velocity (float, optional): Velocity at which q-space is traversed during
            readout. If omitted, the original VSF formulation is used.
        g_background_arr (ndarray, optional): Background magnetic field gradient
            along the specified axis.
        b0_background_arr (ndarray, optional): B0 offset map.
        r2s_arr (ndarray, optional): R2 or R2* relaxation rate map.
        r2p_arr (ndarray, optional): R2' relaxation rate map.

    Returns:
        ndarray of shape ((delta_m_arr.size) + q_background_arr.shape):
            Voxel spread function calculated for each delta_m (first axis) and
            each image location (remaining axes).
    """
    if n_voxels_acq is not None and n_voxels_acq == 1:
        # If the numbers of acquired voxels along the axis is specified and exactly 1, we are in a through-plane dephasing situation.
        # In that case, apply the through-plane estimation of the voxel spread function.
        return calculate_voxel_spread_function_throughplane(q_background_arr, axis, delta_m_arr)
    if q_velocity is None:
        # Use original VSF calculation in case of no supplied q_velocity
        return calculate_original_voxel_spread_function_1axis(q_background_arr, axis, delta_m_arr, n_voxels_acq=n_voxels_acq)
    else:
        # Otherwise use extended VSF calculation with provided values.
        return calculate_extended_voxel_spread_function_1axis(q_background_arr, axis, delta_m_arr, n_voxels_acq=n_voxels_acq,
                                                              q_velocity=q_velocity, g_background_arr=g_background_arr, 
                                                              b0_background_arr=b0_background_arr, r2s_arr=r2s_arr, r2p_arr=r2p_arr)


def precompute_vsf_allm_allaxes(q_background_arr_list, delta_m_arr_list, readout_axis=None, n_voxels_acq_list=None,
                                q_velocity=None, g_background_arr_list=None, b0_background_arr=None, 
                                r2s_arr=None, r2p_arr=None
                                ):
    """Calculates the voxel spread function (eta) along each axis and delta_m and saves them in a big array.
    The dependence on voxel location n is encoded in the array shape, which is taken over from the shape of the
    q_background_arr entries.
    The dependence on voxel location m is encoded as an offset delta_m relative to n.
    The delta_m_arr values should be defined along each axis, by providing them as a list of length q_background_arr_list.
    This re-definition of m seemed to make the implementation more efficient, as we can apply functions to all
    different voxels n at the same time.
    For each axis, either the standard voxel spread function or the through-plane variant is used, depending on
    if more than 1 voxel was acquired along that axis (as specified in n_voxels_acq_list).

    Args:
        q_background_arr_list (list of ndarrays): List of q shifts due to field gradients in unit [rad].
            List should have length equal to the number of image dimensions and each entry should have the same shape
            as the image, defining the q shift in the corresponding direction at each location.
        delta_m_arr_list (list of integer ndarrays): List of same length as q_background_arr_list. Used to define the 
            location m array for all shifts relative to n along each axis.
        n_voxels_acq_list (list of integers, optional): The size of the acquisition matrix along each axis. May differ from 
	        the image matrix due to interpolation. Should ignore parallel imaging or partial fourier at this point.
	        If a given axis is a through-plane axis for 2D imaging, you should choose the value 1 here, as 
            the k-space was only sampled for k=0 along that direction. Defaults to None, meaning acquired voxels equal image shape.
        readout_axis (int, optional): Axis corresponding to the k-space readoutdirection. The extended VSF model is only applied 
            along this axis when q_velocity is supplied.
        q_velocity (float, optional): Velocity at which q-space is traversed during readout. If None, the original VSF model is 
            used along all axes.
        g_background_arr_list (list of ndarrays, optional): Background magnetic field gradient maps for each axis. Must have the 
            same length as q_background_arr_list.
        b0_background_arr (ndarray, optional): B0 map used by the extended VSF model.
        r2s_arr (ndarray, optional): R2 or R2* relaxation rate map used by the extended VSF model.
        r2p_arr (ndarray, optional): R2' relaxation rate map used by the extended VSF model.

    Returns:
        ndarray: Shape [n_axes, n_m] + image.shape, contains the voxel spread function for each axis direction and each
        corresponding delta_m.
    """
    n_axes = len(q_background_arr_list)
    largest_m_number = max([arr.size for arr in delta_m_arr_list])
    per_axis_per_m_vsf_arr_arr = np.zeros(tuple([n_axes, largest_m_number] + list(q_background_arr_list[0].shape)), dtype=COMPLEX_TYPE)

    if g_background_arr_list is None:
        g_background_arr_list = [None]*n_axes
    if q_background_arr_list is None:
        q_background_arr_list = [None]*n_axes
    # Pre-compute the voxel spread function at each delta m along each axis
    for axis in range(n_axes):
        # Only apply the q_velocity along the readout axis
        axis_q_velocity = q_velocity if ((readout_axis is not None) and (axis == readout_axis)) else None
        n_voxels_acq = None if n_voxels_acq_list is None else n_voxels_acq_list[axis]
        n_delta_m = delta_m_arr_list[axis].size
        delta_m_slice = np.arange(n_delta_m, dtype=np.int32)
        print(f"Precomputing VSF for axis {axis+1}/{n_axes}")
        per_axis_per_m_vsf_arr_arr[axis, delta_m_slice] = calculate_voxel_spread_function_1axis(
            q_background_arr_list[axis], axis, delta_m_arr_list[axis], n_voxels_acq=n_voxels_acq,
            q_velocity=axis_q_velocity, g_background_arr=g_background_arr_list[axis], b0_background_arr=b0_background_arr, 
            r2s_arr=r2s_arr, r2p_arr=r2p_arr
            )

    # Print memory size of array, as that can get quite big for big arrays and large delta m.
    print(f"Size of per_axis_per_m_vsf_arr_arr in bytes: {sys.getsizeof(per_axis_per_m_vsf_arr_arr)}")

    return per_axis_per_m_vsf_arr_arr


#########################################
# Functions related to signal summation #
#########################################


def sum_voxel_contributions_allaxes(sigma_arr, phi_background_arr, q_background_arr_list, m_width, readout_axis=None,
                                    n_voxels_acq_list=None, q_velocity=None, g_background_arr_list=None, b0_background_arr=None, 
                                    r2s_arr=None, r2p_arr=None, kspace_mask=None):
    """Implements the summation over all delta_m of the signal contributions,
    as given in Yablonskiy equation [16].

    The dependence on voxel location n is encoded in the array shape, which is
    taken over from sigma_arr. The dependence on voxel location m is encoded as
    an offset delta_m relative to n along each image axis. This re-definition of
    m allows the calculation to be vectorized over all voxel locations n
    simultaneously.

    For each combination of delta_m offsets, the quantity psi_m is assembled
    according to Yablonskiy equation [19] by multiplying the voxel spread
    function contributions from all axes and applying the corresponding phase
    shift. The resulting contribution is multiplied with the shifted source
    signal sigma_m and accumulated into the final result.

    This function pre-computes the voxel spread functions for all axes and
    delta_m values and stores them in a large intermediate array. This improves
    performance but can result in substantial memory usage. For large datasets
    or large m_width values, it may be preferable to process the image in
    sub-regions.

    Args:
        sigma_arr (ndarray): MR image free of artifact-induced signal spreading.
            Should have the shape of the image.
        phi_background_arr (ndarray): Background phase phi in units of [rad].
            Should have the same shape as sigma_arr.
        q_background_arr_list (list of ndarrays): List of q shifts due to field
            gradients in units of [rad]. The list should have length
            phi_background_arr.ndim and each entry should have image shape,
            defining the q shift along the corresponding direction.
        m_width (integer): Number of delta_m steps to evaluate in positive and
            negative directions of each axis. A total of 2*m_width+1 shifts are
            evaluated along each axis, corresponding to
            (2*m_width+1)^n_axes total delta_m combinations.
        readout_axis (int, optional): Axis corresponding to the k-space readout
            direction. The extended VSF model is only applied along this axis
            when q_velocity is supplied.
        n_voxels_acq_list (list of integers, optional): Size of the acquisition
            matrix along each image axis. May differ from the image matrix due
            to interpolation. Parallel imaging and partial Fourier effects are
            ignored. For a through-plane direction in 2D imaging, use a value
            of 1. Defaults to None, meaning acquired voxels equal image shape.
        q_velocity (float, optional): Velocity at which q-space is traversed
            during readout. If None, the original VSF formulation is used.
        g_background_arr_list (list of ndarrays, optional): Background magnetic
            field gradient maps for each axis.
        b0_background_arr (ndarray, optional): B0 offset map used by the
            extended VSF model.
        r2s_arr (ndarray, optional): R2 or R2* relaxation rate map used by the
            extended VSF model.
        r2p_arr (ndarray, optional): R2' relaxation rate map used by the
            extended VSF model.
        kspace_mask (ndarray, optional): K-space filter or sampling mask to be
            applied to the final image after summation.

    Returns:
        ndarray: Image with all voxel signal contributions summed according to
        the voxel spread function model. The output has the same shape as
        sigma_arr.
    """
    exp_phi_background_arr = np.exp(1.0j*phi_background_arr, dtype=COMPLEX_TYPE)

    n_axes = sigma_arr.ndim
    m_width_arr = np.ones(n_axes)*m_width
    if n_voxels_acq_list is not None:
        # For through-plane axes, should not take any delta_m there (to my current understanding)
        m_width_arr[np.array(n_voxels_acq_list) == 1] = 0
    # Define the steps of m we look at
    delta_m_arr_list = [np.arange(-axis_m_width, axis_m_width+1, dtype=np.int32) for axis_m_width in m_width_arr]
    
    per_axis_per_m_vsf_arr_arr = precompute_vsf_allm_allaxes(
        q_background_arr_list, delta_m_arr_list, readout_axis=readout_axis, n_voxels_acq_list=n_voxels_acq_list,
        q_velocity=q_velocity, g_background_arr_list=g_background_arr_list, b0_background_arr=b0_background_arr, 
        r2s_arr=r2s_arr, r2p_arr=r2p_arr,
        )

    index_grids_delta_m = np.meshgrid(*([np.arange(len(arr)) for arr in delta_m_arr_list])) # Define the delta_m_grid in each image axis

    n_points_index_grid = index_grids_delta_m[0].size

    result_arr = np.zeros(sigma_arr.shape, dtype=COMPLEX_TYPE)
    for delta_m_grid_index in range(n_points_index_grid):
        # Construct the delta_m and its index across each axis (for an arbitrary number of axes)
        index_delta_m_list = [grid.ravel()[delta_m_grid_index] for grid in index_grids_delta_m]
        delta_m_list = [delta_m_arr[index_delta_m] for delta_m_arr, index_delta_m in zip(delta_m_arr_list, index_delta_m_list)]
        
        # Assemble psi m like in equation [19] of Yablonskiy
        psi_m = np.ones(sigma_arr.shape, dtype=COMPLEX_TYPE)
        # Multiply the voxel spread functions
        for axis in range(n_axes):
            psi_m *= per_axis_per_m_vsf_arr_arr[axis, index_delta_m_list[axis]]
        psi_m *= shift_arr_allaxes(exp_phi_background_arr, delta_m_list)
        # Multiply psi with sigma to get signal
        sigma_m = shift_arr_allaxes(sigma_arr, delta_m_list)
        result_arr += psi_m*sigma_m
    
    if kspace_mask is not None:
        # Apply mask in k-space to data
        result_arr = apply_kspace_mask(result_arr, kspace_mask)
    
    return result_arr


def calculate_f_factor_similarity_approx_allaxes(s_arr, phi_background_arr, q_background_arr_list, m_width, 
                                                 readout_axis=None,n_voxels_acq_list=None, q_velocity=None, 
                                                 g_background_arr_list=None, b0_background_arr=None, 
                                                 r2s_arr=None, r2p_arr=None, kspace_mask=None):
    """Calculates the f-factor along all image axes using the similarity approximation.

    The f-factor represents the multiplicative signal attenuation caused by
    intravoxel dephasing and voxel spreading, as defined in Yablonskiy
    equation [23]. The calculation uses the similarity approximation from
    equation [21], where the artifact-free signal sigma is approximated by
    the measured signal magnitude.

    The voxel spread function model is evaluated for all relevant voxel
    offsets and used to estimate the signal contributions that would be
    received from neighboring voxels. The resulting signal is normalized by
    the measured signal magnitude to obtain the local attenuation factor.

    Args:
        s_arr (ndarray): Measured MR image from the scanner. May be magnitude
            or complex-valued and can have arbitrary image dimensionality.
        phi_background_arr (ndarray): Background phase phi in units of [rad].
            Should have the same shape as s_arr.
        q_background_arr_list (list of ndarrays): List of q shifts due to field
            gradients in units of [rad]. The list should have length
            phi_background_arr.ndim and each entry should have image shape,
            defining the q shift along the corresponding direction.
        m_width (integer): Number of delta_m steps to evaluate in positive and
            negative directions of each axis. A total of 2*m_width+1 shifts are
            evaluated along each axis, corresponding to
            (2*m_width+1)^n_axes total delta_m combinations.
        readout_axis (int, optional): Axis corresponding to the k-space readout
            direction. The extended VSF model is only applied along this axis
            when q_velocity is supplied.
        n_voxels_acq_list (list of integers, optional): Size of the acquisition
            matrix along each image axis. May differ from the image matrix due
            to interpolation. Parallel imaging and partial Fourier effects are
            ignored. For a through-plane direction in 2D imaging, use a value
            of 1. Defaults to None, meaning acquired voxels equal image shape.
        q_velocity (float, optional): Velocity at which q-space is traversed
            during readout. If None, the original VSF formulation is used.
        g_background_arr_list (list of ndarrays, optional): Background magnetic
            field gradient maps for each axis.
        b0_background_arr (ndarray, optional): B0 offset map used by the
            extended VSF model. Should be scaled by gamma and have unit rad*Hz.
        r2s_arr (ndarray, optional): R2 or R2* relaxation rate map used by the
            extended VSF model.
        r2p_arr (ndarray, optional): R2' relaxation rate map used by the
            extended VSF model.
        kspace_mask (ndarray, optional): k-space filter or sampling mask to be
            applied to the resulting image before calculating the f-factor.

    Returns:
        ndarray: f-factor map with the same shape as the input image. Values
        represent the estimated multiplicative signal attenuation caused by
        dephasing and voxel spreading according to the similarity
        approximation.
    """
    voxel_contributions_arr = sum_voxel_contributions_allaxes(
        np.absolute(s_arr), phi_background_arr, q_background_arr_list, m_width, readout_axis=readout_axis, n_voxels_acq_list=n_voxels_acq_list,
        q_velocity=q_velocity, g_background_arr_list=g_background_arr_list, b0_background_arr=b0_background_arr, r2s_arr=r2s_arr, r2p_arr=r2p_arr,
        kspace_mask=kspace_mask)
    voxel_contributions_arr /= np.absolute(s_arr)
    return np.absolute(voxel_contributions_arr)


#####################################
# Functions to be applied on images #
#####################################


def calculate_f_factor_for_image(image_arr, b0_arr, m_width, t, voxel_size_arr, n_voxels_acq_list=None, 
                                 phi_0_arr=0., readout_axis=None, q_velocity=None,
                                 r2s_arr=None, r2p_arr=None, kspace_mask=None):
    """Calculates the f-factor for a measured MR image.

    The f-factor represents the multiplicative signal attenuation caused by
    intravoxel dephasing and voxel spreading, as defined in Yablonskiy
    equation [23]. The calculation uses the similarity approximation from
    equation [21], where the artifact-free signal sigma is approximated by
    the measured signal magnitude.

    This function serves as a convenience wrapper around
    calculate_f_factor_similarity_approx_allaxes(). It derives the required
    phase map and q-shift maps from the supplied B0 off-resonance map,
    acquisition time, and voxel geometry before performing the f-factor
    calculation.

    By default, the original voxel spread function formulation is used. If
    q_velocity is provided, the extended voxel spread function model is
    used along the specified readout axis, incorporating effects of finite
    k-space traversal speed, B0 evolution, and relaxation during readout.

    Requires an estimate of the phase at t = 0, given by phi_0_arr. This
    may be obtained from a very early echo or acquisition. In many cases,
    assuming phi_0_arr = 0 is sufficient and is therefore the default.

    Args:
        image_arr (ndarray): Measured MR image from the scanner. May be
            magnitude or complex-valued and can have arbitrary image
            dimensionality.
        b0_arr (ndarray): Magnetic field off-resonance map multiplied by the
            gyromagnetic ratio, in units of [rad Hz]. Should have the same
            shape as image_arr.
        m_width (integer): Number of delta_m steps to evaluate in positive
            and negative directions of each axis. A total of
            2*m_width+1 shifts are evaluated along each axis, corresponding
            to (2*m_width+1)^n_axes total delta_m combinations.
        t (scalar): Time since spin refocusing in units of [s]. For GRE,
            this corresponds to time since excitation. For spin-echo based
            sequences, it corresponds to time since the spin echo.
        voxel_size_arr (ndarray): Voxel size in units of [mm] for each image
            axis. Should have length image_arr.ndim.
        n_voxels_acq_list (list of integers, optional): Size of the
            acquisition matrix along each image axis. May differ from the
            image matrix due to interpolation. Parallel imaging and partial
            Fourier effects are ignored. For a through-plane direction in
            2D imaging, use a value of 1. Defaults to None, meaning acquired
            voxels equal image shape.
        phi_0_arr (ndarray or scalar, optional): Phase at t = 0 in units of
            [rad]. May be estimated from an early-time image. Defaults to 0.
        readout_axis (int, optional): Axis corresponding to the k-space
            readout direction. The extended VSF model is only applied along
            this axis when q_velocity is supplied.
        q_velocity (float, optional): Velocity at which q-space is traversed
            during readout. If None, the original VSF formulation is used.
        r2s_arr (ndarray, optional): R2 or R2* relaxation rate map used by
            the extended VSF model.
        r2p_arr (ndarray, optional): R2' relaxation rate map used by the
            extended VSF model.
        kspace_mask (ndarray, optional): K-space filter or sampling mask to
            be applied before calculating the final f-factor.

    Returns:
        ndarray: f-factor map with the same shape as image_arr. Values
        represent the estimated multiplicative signal attenuation caused by
        dephasing and voxel spreading.
    """
    # Estimate background field gradients from the supplied B0 map.
    gx_arr, gy_arr, gz_arr = calculate_central_differences_in_image_axes(
        b0_arr, voxel_size_arr=voxel_size_arr
    )

    if np.all(phi_0_arr == 0.):
        k_0x_arr = k_0y_arr = k_0z_arr = 0.
    else:
        k_0x_arr, k_0y_arr, k_0z_arr = calculate_central_differences_in_image_axes(
            phi_0_arr, voxel_size_arr=voxel_size_arr
        )

    qx_arr = calculate_q(gx_arr, k_0x_arr, t, voxel_size_arr[0])
    qy_arr = calculate_q(gy_arr, k_0y_arr, t, voxel_size_arr[1])
    qz_arr = calculate_q(gz_arr, k_0z_arr, t, voxel_size_arr[2])

    phi_arr = calculate_phi(b0_arr, phi_0_arr, t)

    f_factor_arr = calculate_f_factor_similarity_approx_allaxes(
        image_arr, phi_arr, [qx_arr, qy_arr, qz_arr], m_width, readout_axis=readout_axis,
        n_voxels_acq_list=n_voxels_acq_list, q_velocity=q_velocity, g_background_arr_list=[gx_arr, gy_arr, gz_arr],
        b0_background_arr=b0_arr, r2s_arr=r2s_arr, r2p_arr=r2p_arr, kspace_mask=kspace_mask,
    )

    return f_factor_arr


################################
# Function to test if it works #
################################


def create_gaussian_test_image(nx, ny, nz, gamma_deltab0):
    # Use normal distribution to define a B0
    def normpdf(x, avg=0, std=1):
        exp_term = np.exp(-0.5 * ((x - avg) / std) ** 2)
        pdf = (1 / (std * np.sqrt(2 * np.pi)) * exp_term)
        return pdf
    
    x_arr = np.linspace(-1, 1, nx).reshape((-1, 1, 1))
    y_arr = np.linspace(-1, 1, ny).reshape((1, -1, 1))
    z_arr = np.linspace(-1, 1, nz).reshape((1, 1, -1))

    test_b0_hz = normpdf(x_arr, 0, 0.2)*normpdf(y_arr, 0, 0.2)*normpdf(z_arr, 0, 0.1)
    test_b0_hz *= gamma_deltab0/np.amax(test_b0_hz)

    test_image = (np.absolute(test_b0_hz/gamma_deltab0) > 0.001).astype(np.float64)

    return test_image, test_b0_hz


def test_on_gaussian(nx=64, ny=64, nz=32, gamma_deltab0=600, m_width=10): # config like ASE EPI images
    import matplotlib.pyplot as plt

    print(f"Testing voxel spread function on image of shape ({nx}, {ny}, {nz}) and {2*m_width+1} delta_m values")
    print(f"Using a 3D gaussian B0 field")

    # Define variables
    test_shape = (nx, ny, nz)
    voxel_size_arr = np.array([3.5, 3.5, 2.9]) # ASE EPI
    n_voxels_acq_list = [nx, ny, 1] # ASE EPI

    test_image, test_b0_hz = create_gaussian_test_image(nx, ny, nz, gamma_deltab0)

    # Parameters for extended VSF
    test_r2_arr = 10 * np.ones_like(test_image)
    test_r2p_arr = 4.2 * np.ones_like(test_image)
    readout_axis = 1
    q_velocity = -2 * np.pi / 0.02282 # divided by readout duration in seconds, ASE EPI configuration
    tau = 0.048 # Try just one tau (in s)
    k_0 = 0.
    phi_0 = 0.

    kspace_mask = None

    # Calculate F with original VSF
    f_factor_orig = calculate_f_factor_for_image(
        test_image, test_b0_hz, m_width, tau, voxel_size_arr, 
        n_voxels_acq_list=n_voxels_acq_list, phi_0_arr=phi_0, 
        kspace_mask=None
    )
    # Calculate F with extended VSF
    f_factor_ext = calculate_f_factor_for_image(
        test_image, test_b0_hz, m_width, tau, voxel_size_arr, 
        n_voxels_acq_list=n_voxels_acq_list, phi_0_arr=phi_0, readout_axis=readout_axis, q_velocity=q_velocity, 
        r2s_arr=test_r2_arr, r2p_arr=test_r2p_arr, 
        kspace_mask=None
    )
    
    # Plot
    plot_slice_index = int(test_shape[2]/2)
    fig, axes = plt.subplots(1, 3, squeeze=True, layout="constrained", figsize=(3*2.1, 2.4), dpi=100)

    axes[0].imshow(test_b0_hz[:,:,plot_slice_index], interpolation='none', 
                   vmin=-400, vmax=400, cmap="PRGn")
    axes[0].set_title("B0 map")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(f_factor_orig[:,:,plot_slice_index], interpolation='none', 
                   vmin=0, vmax=1.3, cmap="hot")
    axes[1].set_title("F from original VSF")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[2].imshow(f_factor_ext[:,:,plot_slice_index], interpolation='none', 
                   vmin=0, vmax=1.3, cmap="hot")
    axes[2].set_title("F from extended VSF")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.show()

    print("Done")

if __name__ == "__main__":
    test_on_gaussian(nx=64, ny=64, nz=32, m_width=10) # config like ASE EPI image