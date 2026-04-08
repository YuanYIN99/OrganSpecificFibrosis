# This script contains all subfunctions used in 'migration_main.py'
# MODEL assumptions to ask: search 'CHECK'

import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm 


def cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool):
    ''' 
    This function finds the pairwise distance and orientation bewtween cells.
    INPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'y_len': height of the rectangular domain
    'x_len': width of the rectangular domain
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction 
    OUTPUT:
    'dist': a 2D np symmetrix array of size N*N storing pairwise cell distance
    'orientation': a 2D np array of size N*N storing pairwise cell orientation

    DEBUGGED
    '''

    # pairwise distance in x and y directions:
    dx = cell_coords[:, np.newaxis][:, :, 0] - cell_coords[:, 0]
    dy = cell_coords[:, np.newaxis][:, :, 1] - cell_coords[:, 1]

    # y-periodicity if specified:
    if y_periodic_bool: 
        dy_abs = abs(dy)
        dy_H = y_len - dy_abs
        mask1 = (dy_abs >= dy_H) & (cell_coords[:, 1] < y_len/2)
        dy[mask1] = - dy_H[mask1]
        mask2 = (dy_abs >= dy_H) & (cell_coords[:, 1] >= y_len/2)
        dy[mask2] = dy_H[mask2]

    # x-periodicity if specified:
    if x_periodic_bool: 
        dx_abs = abs(dx)
        dx_H = x_len - dx_abs
        mask1 = (dx_abs >= dx_H) & (cell_coords[:, 0] < x_len/2)
        dx[mask1] = - dx_H[mask1]
        mask2 = (dx_abs >= dx_H) & (cell_coords[:, 0] >= x_len/2)
        dx[mask2] = dx_H[mask2]

    dist = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) 

    # for cells overlapping, randomly select its migrating orientation:
    overlap_mask = (dx == 0.0) & (dy == 0.0) 
    orientation[overlap_mask] = np.random.uniform(-np.pi, np.pi, len(orientation[overlap_mask]))

    return dist, orientation


def total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t):
    ''' 
    This function calculates the inter-cellular forces for each cell based on their pairwise distances and orientations
    INPUT:
    'dist': a 2D np symmetrix array of size N*N storing pairwise cell distance
    'orientation': a 2D np array of size N*N storing pairwise cell orientation
    'sigma': the constant cell diameter uniform across the cell population
    'epsilon': depth of the Lennard Jone's potential 
    'r_max': the finate range of cell-cell interactions
    'rep_adh_len': the characteristic length representing the balance between repulsion and adhesion
    OUTPUT:
    'F_cc': a 2D np array of size N*2 storing the inter-cellular forces experienced by each cell
    'total_adh_magnitude': a 1D np array of size N storing the magnitudes of total adhesion force felt by cells
    'rep_dist_': a 2D np array of size N*N storing the distance to 'rep_adh_len' for repulsive neighbours
    'num_rep_neigh': a 1D np array of size N storing the number of repulsive neighbours
    'num_adh_neigh': a 1D np array of size N storing the number of adhesive neighbours

    DEBUGGED
    '''

    # initialise the magnitude of forces exposed by neighbouring cells for all the N cells:
    cc_magnitude = np.zeros((len(dist), len(dist)), dtype=float)
    
    # calculate the magnitude of forces based on the Lennard Jone's potential:
    mask = dist <= r_max
    np.fill_diagonal(dist, np.nan) # no inter-cellular force on a cell due to the presence of itself
    cc_magnitude[mask] = (2 * (sigma**6)/(dist[mask]**6) - 1) * 24 * epsilon * (sigma**6) / (dist[mask]**7)
    # for numerical stability, if abs(cc_magnitude) > sigma/2, then manually set abs(vel_t) = sigma/2
    too_large_mask, too_small_mask = cc_magnitude > 0.2*sigma, cc_magnitude < -0.2*sigma
    cc_magnitude[too_large_mask], cc_magnitude[too_small_mask] = 0.2*sigma, -0.2*sigma

    # obtain the orientation, thus the vector representation, of forces:
    # x-direction: 
    cc_dx_cells = np.multiply(cc_magnitude, np.cos(orientation))
    np.fill_diagonal(cc_dx_cells, np.nan) 
    cc_dxpart = np.nansum(cc_dx_cells, axis=1) # sum over all its neighbouring cells
    # y-direction:
    cc_dy_cells = np.multiply(cc_magnitude, np.sin(orientation))
    np.fill_diagonal(cc_dy_cells, np.nan)
    cc_dypart = np.nansum(cc_dy_cells, axis=1) 

    F_cc = np.column_stack([cc_dxpart, cc_dypart])

    # deal with values that are super close to zero (to avoid floating point error):
    small_mask = np.abs(F_cc) <= 10**(-10)
    F_cc[small_mask] = 0.0

    # count the number of repulsive and adhesive neighbours:
    rep_mask, adh_mask = dist<=rep_adh_len, (dist>rep_adh_len)&(dist <= r_max)
    num_rep_neigh, num_adh_neigh = np.nansum(rep_mask,axis=1), np.nansum(adh_mask,axis=1)
    # the 'np.nansum' in calculating 'num_rep_neigh' is because: a cell isn't itself's repulsive neighbour.

    # for the usage in 'CGcc_Pc': --------------------------------------------------------
    # output the total adhesion forces felt by cells:
    adh_magnitude = np.zeros((len(dist), len(dist)), dtype=float)
    adh_magnitude[adh_mask] = (2 * (sigma**6) / (dist[adh_mask]**6) - 1) * 24 * epsilon * (sigma**6) / (dist[adh_mask]**7)
    total_adh_magnitude = np.nansum(adh_magnitude, axis=1)
    # we don't need to care about the cell to itself, as the distance to itself is zero, falling outside the adhesive range.

    # output the distance to 'rep_adh_len' for repulsive neighbouring cells:
    rep_dist_ = np.zeros((len(dist), len(dist)), dtype=float)
    rep_dist_[rep_mask] = 1 - dist[rep_mask]/rep_adh_len
    np.fill_diagonal(rep_dist_, 0)
    # -------------------------------------------------------------------------------------

    return F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh


def total_F_rm(D, n, delta_t, total_fibre_cell_loc, D_min):
    '''
    This function samples from a 2D white noise and obtains random motion of with macroscopic coefficient 'D' for the cell population.
    INPUT:
    'D': the macroscopic coefficient
    'n': number of cells at time t
    'delta_t': numerical time step 
    ### YY: extended model ###
    'total_fibre_cell_loc': a 1D np array of size N storing total fibre space-filling percentages at cell locations
    'D_min': minimum diffusion coefficient
    ########################################
    OUTPUT:
    'F_rm': a 2D np array storing random forces experienced by each cell.

    DEBUGGED
    '''
    ### YY: extended model ###
    D_effective = D - (D - D_min) * total_fibre_cell_loc
    F_rm = np.array([
        np.random.multivariate_normal(
            mean=np.zeros(2), 
            cov=2 * D_i * np.eye(2) / delta_t
        ) 
        for D_i in D_effective
    ])
    ########################################
    #F_rm = np.random.multivariate_normal(mean=np.zeros(2), cov=2*D*np.eye(2)/delta_t, size=n)

    return F_rm


def fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N):
    '''
    This function interpolates the fibre field 'Omega' at cell central locations, it also returns the 
    total fibre space-filling percentages at those locations.
    INPUT:
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'Omega': a 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'N': cell population at time t

    OUTPUT:
    'Omega_cell_loc': a 3D np array of size (N, 2, 2) storing fibre tensor at each cell location
    'total_fibre_cell_loc': a 1D np array of size N storing total fibre space-filling percentages at cell locations

    DEBUGGED
    '''
    # create spline interpolation objects for the 00, 01, 10, and 11 entries of Omega on the grid defined:
    # (we set kx=ky=1 for bilinear interpolation, so that there is no impact on sign of the value interpolated)
    interpolator_00 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 0, 0].T, kx=1, ky=1)
    interpolator_01 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 0, 1].T, kx=1, ky=1)
    interpolator_10 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 1, 0].T, kx=1, ky=1)
    interpolator_11 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 1, 1].T, kx=1, ky=1)

    # interpolate the 00, 01, 10, and 11 entries of Omega at the cell locations:
    # (we set 'grid=False' to interpolate values at these points, rather than on the grid defined by these points)
    Omega_00_interp = interpolator_00(cell_coords[:, 0], cell_coords[:, 1], grid=False)
    Omega_01_interp = interpolator_01(cell_coords[:, 0], cell_coords[:, 1], grid=False)
    Omega_10_interp = interpolator_10(cell_coords[:, 0], cell_coords[:, 1], grid=False)
    Omega_11_interp = interpolator_11(cell_coords[:, 0], cell_coords[:, 1], grid=False)

    # stack and resize:
    Omega_cell_loc = np.stack((Omega_00_interp, Omega_01_interp, Omega_10_interp, Omega_11_interp), axis=-1)
    Omega_cell_loc = Omega_cell_loc.reshape(N, 2, 2)
    
    total_fibre_cell_loc = np.trace(Omega_cell_loc, axis1=1, axis2=2)

    return Omega_cell_loc, total_fibre_cell_loc


# Define the custom hyperbolic tanh function
def scaled_tanh(x, shift, scale):
    # Hyperbolic tanh, shifted and scaled
    return 0.5 * (np.tanh(scale * (x - shift)) + 1)
# Adjust the function to ensure it starts at 0 and ends at 1
def adjusted_tanh(x, shift, scale):
    tanh_value = scaled_tanh(x, shift, scale)
    # Adjusting so that the curve starts at 0 at x=0 and reaches 1 at x=1
    return (tanh_value - scaled_tanh(0, shift, scale)) / (scaled_tanh(1, shift, scale) - scaled_tanh(0, shift, scale))

def CG_rand(total_fibre_cell_loc, F_rm, Omega_cell_loc, N, shift, scale, n):
    '''
    This function calculates the fibre contact guidance matrix associated with random motion for each cell. 
    INPUT:
    'total_fibre_cell_loc': a 1D np array of size N storing total fibre space-filling percentages at cell locations
    'F_rm': a 2D np array storing random forces experienced by each cell
    'Omega_cell_loc': a 3D np array of size (N, 2, 2) storing fibre tensor at each cell location
    'N': cell population at time t
    OUTPUT:
    'M_rand_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations
    
    DEBUGGED
    '''

    # find the length-preserving Omega at cell locations:
    Omega_hat = np.zeros((N, 2, 2), dtype = float) 
    l_rand = norm(np.squeeze(Omega_cell_loc@F_rm[:,:,np.newaxis], axis=2), axis=1) # matrix vector multiplication length
    lnonzero_mask = np.abs(l_rand) > 10**(-10) # we only care about non-zero motility vector after contact guidance
    Omega_hat[lnonzero_mask] = (Omega_cell_loc[lnonzero_mask]) * ((norm(F_rm,axis=1)[lnonzero_mask]/l_rand[lnonzero_mask])[:,np.newaxis,np.newaxis]) # normalise
    # When F_rm is in the direction of the v2 but lambda2=0 (thus CG will modulate velocity vectors into points)
    zero_mask = np.logical_and(np.abs(l_rand)<=10**(-10), norm(F_rm,axis=1)>10**(-10)) 
    zero_mask = np.logical_and(zero_mask, np.any(np.abs(Omega_cell_loc)>10**(-10), axis=(1, 2))) # there is fibres
    zero_indices = np.where(zero_mask)[0]
    # the strength of contact guidance on a cell's random motion is weighted by the space-filling degree of fibres at the cell location:
    # CG's strength is linearly dependent on total area density (NOT OBVIOUS RESULTS)
    #M_rand = total_fibre_cell_loc[:,np.newaxis,np.newaxis] * Omega_hat + \
    #    (1-total_fibre_cell_loc[:,np.newaxis,np.newaxis]) * np.eye(2, dtype=float)
    # TRY NONLINEAR DEPENDENCE (Tanh):
    Lambda = adjusted_tanh(total_fibre_cell_loc, shift, scale)
    zero_CG_strength = Lambda[zero_indices]
    M_rand = Lambda[:,np.newaxis,np.newaxis]*Omega_hat + (1-Lambda[:,np.newaxis,np.newaxis])*np.eye(2, dtype=float)

    # normalise 'M_rand':
    M_rand_hat = np.zeros((N, 2, 2), dtype = float) 
    l_M = norm(np.squeeze(M_rand@F_rm[:,:,np.newaxis], axis=2), axis=1)
    lMnonzero_mask = np.abs(l_M) > 10**(-10)
    M_rand_hat[lMnonzero_mask] = M_rand[lMnonzero_mask] * ((norm(F_rm,axis=1)[lMnonzero_mask]/l_M[lMnonzero_mask])[:,np.newaxis,np.newaxis])
    # for 'zero_indices': set M_rand_hat = zero matrix as we are going to tackle this scenario in 'migrate_t'
    M_rand_hat[zero_indices] = np.zeros(2, dtype=float)

    return M_rand_hat, zero_indices, zero_CG_strength


def CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y):
    '''
    This function calculates the relative fibre distributions P_f for all the cells 
    INPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'N': cell population at time t
    'y_len': height of the rectangular domain
    'x_len': width of the rectangular domain
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction 
    'Omega': a 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'fibre_coords': a 2D np array of size (num_col*num_row, 2) storing coordinates of all the fibre grid points
    OUTPUT:
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'total_fibre_density': a 1D np array of length num_col*num_row storing lambda_1+lambda_2 values on the grid points defined
    'P_f': a 1D np array of length N storing the relative fibre distributions P_f for all N cells  

    DEBUGGED
    '''

    # for each cell, find the pairwise distances to all the fibre grid points:
    dx = cell_coords[:, np.newaxis][:, :, 0]-fibre_coords[:, 0]
    dy = cell_coords[:, np.newaxis][:, :, 1]-fibre_coords[:, 1]

    # y-periodicity if specified:
    if y_periodic_bool: 
        dy_abs = abs(dy)
        dy_H = y_len - dy_abs
        mask1 = (dy_abs >= dy_H) & (fibre_coords[:, 1] < y_len/2)
        dy[mask1] = - dy_H[mask1]
        mask2 = (dy_abs >= dy_H) & (fibre_coords[:, 1] >= y_len/2)
        dy[mask2] = dy_H[mask2]

    # x-periodicity if specified:
    if x_periodic_bool: 
        dx_abs = abs(dx)
        dx_H = x_len - dx_abs
        mask1 = (dx_abs >= dx_H) & (fibre_coords[:, 0] < x_len/2)
        dx[mask1] = - dx_H[mask1]
        mask2 = (dx_abs >= dx_H) & (fibre_coords[:, 0] >= x_len/2)
        dx[mask2] = dx_H[mask2]

    cf_dist = np.sqrt(dx**2 + dy**2) # row i stores cell i's distances to all the grid points 
    #contrib_gridpts_mask = cf_dist <= r_max # all the grid points lying within 'r_max' are included when calculating the fibre CG contribution to inter-cellular interactions

    # caclulate the total fibre space-filling percentage (i.e. lambda_1+lambda_1) on the grid points:
    Omega_reshape = np.reshape(Omega, (len(grid_x)*len(grid_y), 2, 2))
    total_fibre_density = np.trace(Omega_reshape, axis1=1, axis2=2)
    #contrib_fibre_density = [total_fibre_density[row] for row in contrib_gridpts_mask] # fibre densities contributing to P_f

    # carrying capacity of fibres in a cell's neighbourhood:
    #carrying_f = np.pi * (r_max**2)

    # calculate the P_f for all the cells:
    #P_f = np.zeros(N, dtype=float)
    #for i in range(N): 
    #    # we approximate the integration in the numerator by area*(mean of values defined in the region of integration)
    #    P_f[i] = (np.mean(contrib_fibre_density[i])*np.pi*(r_max**2)) / carrying_f 
   
    return cf_dist, Omega_reshape, total_fibre_density #, P_f


def CGcc_Pc(F_adh_max, total_adh_magnitude, rep_dist_):
    '''
    This function calculates the relative inter-cellular force-weighted cell distribution.
    INPUT:
    'F_adh_max': a parameter representing the maximum adhesive (shall be the most negative) forces based on the Lennard Jone's potential
    'total_adh_magnitude': a 1D np array of length N storing the magnitudes of total adhesion force felt by cells
    'rep_dist_': a 2D np array of size N*N storing the distance to 'rep_adh_len' for repulsive neighbours
    OUTPUT:
    'P_c': a 1D np array of length N storing the relative inter-cellular force-weighted cell distribution P_c for all N cells  

    DEBUGGED
    '''

    # calculate the repulsion-weighted and the adhesion-weighted cell distributions:
    rep_distri = (1/6) * np.nansum(rep_dist_, axis=1)
    adh_distri = (1/30) * total_adh_magnitude / F_adh_max

    P_c = rep_distri + adh_distri

    return P_c


def CG_cc(beta, F_cc, P_c, P_f, N, Omega_cell_loc):
    '''
    This function calculates the fibre contact guidance matrix associated with inter-cellulat interactions for each cell.
    INPUT:
    'beta': Hill's coefficient controlling the switch-like behaviour of contact guidance
    'F_cc': a 2D np array of size N*2 storing the inter-cellular forces experienced by each cell
    'P_c': a 1D np array of length N storing the relative inter-cellular force-weighted cell distribution P_c for all N cells 
    'P_f': a 1D np array of length N storing the relative fibre distributions P_f for all N cells 
    'N': cell population at time t 
    'Omega_cell_loc': a 3D np array of size (N, 2, 2) storing fibre tensor at each cell location
    OUTPUT:
    'M_cc_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations

    DEBUGGED
    '''

    # find the length-preserving Omega at cell locations:
    Omega_hat = np.zeros((N, 2, 2), dtype = float) 
    l_rand = norm(np.squeeze(Omega_cell_loc@F_cc[:,:,np.newaxis], axis=2), axis=1) # matrix vector multiplication length
    lnonzero_mask = l_rand > 0.0 # we only care about non-zero motility vector after contact guidance
    Omega_hat[lnonzero_mask] = Omega_cell_loc[lnonzero_mask] * ((norm(F_cc,axis=1)[lnonzero_mask]/l_rand[lnonzero_mask])[:,np.newaxis,np.newaxis]) # normalise

    # the strength of contact guidance on a cell's inter-cellular motion is weighted by the balance between the fibre distribuiton and the cell distribuiton in its vicinity
    M_cc = (P_f**beta/(P_c**beta+P_f**beta))[:,np.newaxis,np.newaxis] * Omega_hat + \
        (P_c**beta/(P_c**beta+P_f**beta))[:,np.newaxis,np.newaxis] * np.full((N,2,2), np.array([[1.0,0.0],[0.0,1.0]]))
    
    # normalise 'M_rand':
    M_cc_hat = np.zeros((N, 2, 2), dtype = float) 
    l_M = norm(np.squeeze(M_cc@F_cc[:,:,np.newaxis], axis=2), axis=1)
    lMnonzero_mask = l_M > 0.0
    M_cc_hat[lMnonzero_mask] = M_cc[lMnonzero_mask] * ((norm(F_cc,axis=1)[lMnonzero_mask]/l_M[lMnonzero_mask])[:,np.newaxis,np.newaxis])

    return M_cc_hat


def cellpool_confluency(pool_top, cell_coords, cell_pool_width, sigma, hstripe_N, y_max, x_min, x_max, y_min, topbool, bottombool):
    # hstripe_N: number cells in conflency in a horizontal stripe of length x_max-x_min and width cell_pool_width

    num_hstripe = int(cell_pool_width/sigma)

    added_cellcoords = []
    remove_cellindices = []
    for i in range(num_hstripe): # loop through each horizontal stripe
        # for top pool:
        if topbool: 
            cells_top_i = np.logical_and(cell_coords[:,1]>=pool_top+i*sigma, cell_coords[:,1]<pool_top+(i+1)*sigma)
            cells_top_i_index = np.where(cells_top_i == True)[0]
            cellnum_top_i = np.sum(cells_top_i)
            if cellnum_top_i < hstripe_N: # need to add more cells for confluency state
                addnum_top_i = int(hstripe_N - cellnum_top_i)
                add_xcoord_top_i = np.random.uniform(x_min, x_max, addnum_top_i)
                add_ycoord_top_i = np.random.uniform(pool_top+i*sigma, pool_top+(i+1)*sigma, addnum_top_i)
                added_cellcoords.extend(list(np.column_stack((add_xcoord_top_i, add_ycoord_top_i))))
            if cellnum_top_i > hstripe_N: # need to remove cells!
                removenum_top_i = int(cellnum_top_i - hstripe_N)
                remove_indice_top_i = np.random.choice(cells_top_i_index, removenum_top_i, replace=False)
                remove_cellindices.extend(list(remove_indice_top_i))

        # for bottom pool:
        if bottombool: 
            cells_bottom_i = np.logical_and(cell_coords[:,1]>=y_min+i*sigma, cell_coords[:,1]<y_min+(i+1)*sigma)
            cells_bottom_i_index = np.where(cells_bottom_i == True)[0]
            cellnum_bottom_i = np.sum(cells_bottom_i)
            if cellnum_bottom_i < hstripe_N: # need to add more cells for confluency state
                addnum_bottom_i = int(hstripe_N - cellnum_bottom_i)
                add_xcoord_bottom_i = np.random.uniform(x_min, x_max, addnum_bottom_i)
                add_ycoord_bottom_i = np.random.uniform(y_min+i*sigma, y_min+(i+1)*sigma, addnum_bottom_i)
                added_cellcoords.extend(list(np.column_stack((add_xcoord_bottom_i, add_ycoord_bottom_i))))
            if cellnum_bottom_i > hstripe_N: # need to remove cells!
                removenum_bottom_i = int(cellnum_bottom_i - hstripe_N)
                remove_indice_bottom_i = np.random.choice(cells_bottom_i_index, removenum_bottom_i, replace=False)
                remove_cellindices.extend(list(remove_indice_bottom_i))

    added_cellcoords = np.array(added_cellcoords)

    return added_cellcoords, remove_cellindices


def cell_proliferation(Delta_0, rho_0, xi, prolif_bool, cell_coords, num_rep_neigh, num_adh_neigh, N, y_periodic_bool, x_periodic_bool, \
                       y_len, x_len, x_min, x_max, y_min, y_max, delta_t, growth_rate): 
    '''
    This function allows a simple density-dependent proliferation.
    INPUT:
    'Delta_0': a parameter denoting the division time when cell i is by itself
    'rho_0': an integer representing the carrying number of cells in a cell's neighbourhood area
    'xi': a parameter denoting the distance a child cell that will be placed from its mother cell
    'prolif_bool': a boolean variable determining whether to allow proliferation in our model
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'num_rep_neigh': a 1D np array of size N storing the number of repulsive neighbours
    'num_adh_neigh': a 1D np array of size N storing the number of adhesive neighbours
    'N': the cell population at time N
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction 
    'y_len': height of the rectangular domain, = 'y_max' - 'y_min'
    'x_len': width of the rectangular domain, ='x_max' - 'x_min'
    OUTPUT:
    'cell_coords': an updated 2D np array of size >=N*2 for >= N cells

    DEBUGGED
    '''
    
    num_rep_neigh = num_rep_neigh.astype(float) # make sure that the data type is float as required.

    if prolif_bool: # if proliferation is allowed
        # calculate the mean division times for all the cells:
        #divi_T =  Delta_0 * (1 + ((num_rep_neigh+num_adh_neigh)/rho_0)**4) 
        #divi_probs = delta_t / divi_T # this uses linear approximation for small delta_t
        # FOR ACCURATE PROB:
        #divi_probs = 1 - np.exp(-delta_t / (Delta_0 * (1.0 + num_rep_neigh/rho_0)))
        # FOR Simpler PROB:
        divi_probs = (delta_t/Delta_0) * (1-num_rep_neigh/rho_0)
        # FOR SIMPLEST EXP GROWTH MODEL: 
        #divi_probs = growth_rate * delta_t

        # sample random numbers from U[0, 1] to determine whether the cell is able to prolifer  ate:
        u = np.random.rand(N)
        prolif_mask = np.logical_and(u<=divi_probs, num_rep_neigh<rho_0) # avoid over-crowdedness
        mother_coord = cell_coords[prolif_mask]

        # place child cells at a distance 'xi' to the mother cells at random angles
        child_angles = np.random.uniform(low=0.0, high=2*np.pi, size=np.sum(prolif_mask))
        child_dx, child_dy = np.cos(child_angles), np.sin(child_angles)
        child_coords = mother_coord + xi*np.column_stack((child_dx, child_dy))
        
        # update child_coords if periodic boundary conditions:
        # (note we do not include the children cells if falling outside the domain of interest given no periodic boundary conditions)
        # in x direction:
        outside_x = np.logical_or(child_coords[:, 0]<x_min, child_coords[:, 0]>x_max)
        if x_periodic_bool:
            child_coords[outside_x, 0] = child_coords[outside_x, 0] % x_len
        else:
            child_coords = child_coords[~outside_x]
        # in y direction:
        outside_y = np.logical_or(child_coords[:, 1]<y_min, child_coords[:, 1]>y_max)
        if y_periodic_bool:
            child_coords[outside_y, 1] = child_coords[outside_y, 1] % y_len
        else:
            child_coords = child_coords[~outside_y]
    else:
        mother_coord = np.empty(0)

    return child_coords, mother_coord


def cf_weight(omega_0, sigma, cf_dist_i):
    '''
    This function calculates the cell-fibre weight function 'omega_cf' to encapsulate the local non-uniform feedback from cells to the surrounding fibres.
    INPUT:
    'omega_0': a parameter determining the maximum cell-fibre effect
    'sigma': the constant cell diameter uniform across the cell population
    'cf_dist_i': a 1D np array of length m storing the pairwise distance between cell i and the grid points lying within its fibre-cell impact radius
    OUTPUT:
    'weight_i': a 1D np array of length m storing the weight cell i imposed on all the grid points within its fibre-cell impact radius

    DEBUGGED
    '''

    weight_i = omega_0 * (1 - cf_dist_i / (sigma/2))

    # no contributions on fibres if the fibre grid points are outside cell i's cell-fibre impact radius:
    outside_mask = cf_dist_i > sigma/2
    weight_i[outside_mask] = 0.0

    return weight_i


def fibre_degradation(Omega_reshape, cf_dist, sigma, d, omega_0, N, grid_x, grid_y, delta_t):
    '''
    This function updates the fibre field Omega due to cell degradations.
    INPUT:
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'sigma': the constant cell diameter uniform across the cell population
    'd': the parameter denoting the constant degradation rate
    'omega_0': a parameter determining the maximum cell-fibre effect
    'N': cell population at time t
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'delta_t': a fixed time stepping size
    OUPTUT:
    'Omega': an updated (due to fibre degradation by cells) 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point

    DEBUGGED
    '''

    # find the fibre grid points lying in a cell's impact area on fibres:
    contrib_gridpts_mask = cf_dist <= sigma/2
    contrib_fibre_dist = [row[mask_row] for row, mask_row in zip(cf_dist, contrib_gridpts_mask)] # row i: contributing fibre grid points for cell i

    # loop through each cell and degrate fibres:
    for i in range(N):
        weight_i = cf_weight(omega_0, sigma, contrib_fibre_dist[i])
        deg = weight_i[:,np.newaxis,np.newaxis] * d * Omega_reshape[contrib_gridpts_mask[i]] * delta_t
        Omega_reshape[contrib_gridpts_mask[i]] = Omega_reshape[contrib_gridpts_mask[i]] - deg
    
    # shape back: 
    Omega = np.reshape(Omega_reshape, (len(grid_y), len(grid_x), 2, 2))

    return Omega, Omega_reshape


def fibre_secretion(Vel_cells, total_fibre_density, Omega_reshape, cf_dist, sigma, s, omega_0, N, grid_x, grid_y, delta_t, tau, num_rep_neigh):
    '''
    This function updates the fibre field Omega due to cell secretions.
    INPUT:
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and it stores cell i's current and all the previous velocities
    'total_fibre_density': a 1D np array of length num_col*num_row storing lambda_1+lambda_2 values on the grid points defined
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'sigma': the constant cell diameter uniform across the cell population
    's': the parameter denoting the constant secretion rate
    'omega_0': a parameter determining the maximum cell-fibre effect
    'N': cell population at time t
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'delta_t': a fixed time stepping size
    'tau': the memory time length when calculating a cell's average migratory direction (this is the direction of the newly laid down fibres)
    ##### YY: extended model #####
    'num_rep_neigh': a 1D np array of size N storing the number of repulsive neighbours
    OUPTUT:
    'Omega': an updated (due to fibre secretion by cells) 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    
    DEBUGGED
    '''

    num_rep_neigh = num_rep_neigh.astype(float) # make sure that the data type is float as required.

    # find the fibre grid points lying in a cell's impact area on fibres:
    contrib_gridpts_mask = cf_dist <= sigma/2
    contrib_fibre_dist = [row[mask_row] for row, mask_row in zip(cf_dist, contrib_gridpts_mask)] # row i: contributing fibre grid points for cell i
    contrib_fibre_density = [total_fibre_density[row] for row in contrib_gridpts_mask] # fibre densities contributing to cell secretion

    # loop through each cell and secrete fibres:
    for i in range(N):
        # find the average velocity (including the current step) in the past tau times, thus 'omega_sec':
        m = int(tau / delta_t)
        prev_vel_i = Vel_cells[i]
        if len(prev_vel_i) < m: # if the existing time for cell i has not reached tau: average over the cell's current age (which is smaller than m)
            vel_ave_i = np.mean(np.vstack(prev_vel_i), axis=0)
        else:
            vel_ave_i = np.mean(np.vstack(prev_vel_i[-m:]), axis=0)
        vel_ave_l = norm(vel_ave_i)
        if vel_ave_l > 0:  # make this unit length
            unit_vel_ave_i = vel_ave_i / vel_ave_l
            omega_sec = np.outer(unit_vel_ave_i, unit_vel_ave_i)
        else: # if the average velocity in the past tau times is zero, then produce isotropic fibres
            omega_sec = np.array([[1.0, 0.0], [0.0, 1.0]])

        weight_i = cf_weight(omega_0, sigma, contrib_fibre_dist[i])
        ### YY: extended model ###: s linearly depends on the number of repulsive neighbours
        effective_s = s * (num_rep_neigh[i] / 6)
        #sec = (weight_i*(1-contrib_fibre_density[i]))[:,np.newaxis,np.newaxis] * s * omega_sec * delta_t
        sec = (weight_i*(1-contrib_fibre_density[i]))[:,np.newaxis,np.newaxis] * effective_s * omega_sec * delta_t
        ##############################
        Omega_reshape[contrib_gridpts_mask[i]] = Omega_reshape[contrib_gridpts_mask[i]] + sec
    
    # shape back: 
    Omega = np.reshape(Omega_reshape, (len(grid_y), len(grid_x), 2, 2))

    return Omega, Omega_reshape


def fibre_secretion_CONST(Vel_cells, total_fibre_density, Omega_reshape, cf_dist, sigma, s, omega_0, N, grid_x, grid_y, delta_t, tau, num_rep_neigh):
    '''
    This function updates the fibre field Omega due to cell secretions.
    INPUT:
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and it stores cell i's current and all the previous velocities
    'total_fibre_density': a 1D np array of length num_col*num_row storing lambda_1+lambda_2 values on the grid points defined
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'sigma': the constant cell diameter uniform across the cell population
    's': the parameter denoting the constant secretion rate
    'omega_0': a parameter determining the maximum cell-fibre effect
    'N': cell population at time t
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'delta_t': a fixed time stepping size
    'tau': the memory time length when calculating a cell's average migratory direction (this is the direction of the newly laid down fibres)
    ##### YY: extended model #####
    'num_rep_neigh': a 1D np array of size N storing the number of repulsive neighbours
    OUPTUT:
    'Omega': an updated (due to fibre secretion by cells) 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    
    DEBUGGED
    '''

    num_rep_neigh = num_rep_neigh.astype(float) # make sure that the data type is float as required.

    # find the fibre grid points lying in a cell's impact area on fibres:
    contrib_gridpts_mask = cf_dist <= sigma/2
    contrib_fibre_dist = [row[mask_row] for row, mask_row in zip(cf_dist, contrib_gridpts_mask)] # row i: contributing fibre grid points for cell i
    contrib_fibre_density = [total_fibre_density[row] for row in contrib_gridpts_mask] # fibre densities contributing to cell secretion

    # loop through each cell and secrete fibres:
    for i in range(N):
        # find the average velocity (including the current step) in the past tau times, thus 'omega_sec':
        m = int(tau / delta_t)
        prev_vel_i = Vel_cells[i]
        if len(prev_vel_i) < m: # if the existing time for cell i has not reached tau: average over the cell's current age (which is smaller than m)
            vel_ave_i = np.mean(np.vstack(prev_vel_i), axis=0)
        else:
            vel_ave_i = np.mean(np.vstack(prev_vel_i[-m:]), axis=0)
        vel_ave_l = norm(vel_ave_i)
        if vel_ave_l > 0:  # make this unit length
            unit_vel_ave_i = vel_ave_i / vel_ave_l
            omega_sec = np.outer(unit_vel_ave_i, unit_vel_ave_i)
        else: # if the average velocity in the past tau times is zero, then produce isotropic fibres
            omega_sec = np.array([[1.0, 0.0], [0.0, 1.0]])

        weight_i = cf_weight(omega_0, sigma, contrib_fibre_dist[i])
        sec = (weight_i*(1-contrib_fibre_density[i]))[:,np.newaxis,np.newaxis] * s * omega_sec * delta_t
        Omega_reshape[contrib_gridpts_mask[i]] = Omega_reshape[contrib_gridpts_mask[i]] + sec
    
    # shape back: 
    Omega = np.reshape(Omega_reshape, (len(grid_y), len(grid_x), 2, 2))

    return Omega, Omega_reshape


def migrate_t(cell_coords, Vel_cells, M_cc_hat, M_rand_hat, F_rm, F_cc, eta, delta_t, y_periodic_bool, x_periodic_bool, x_min, x_max, x_len, y_min, y_max, y_len, N, 
              zero_indices_cc, zero_CG_strengt_cc, zero_indices_rm, zero_CG_strength_rm, Omega_cell_loc): 
    '''
    This function migrates based on force, CG, and the boundary information. However, we are not removing cells if they migrate outside the domain in this function (will be done later). 
    INPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and it stores cell i's current and all the previous velocities
    'M_cc_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations
    'M_rand_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations
    'F_rm': a 2D np array storing random forces experienced by each cell
    'F_cc': a 2D np array of size N*2 storing the inter-cellular forces experienced by each cell
    'eta': cell's effective friction coefficient
    'delta_t': a fixed time stepping size
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction
    'y_len': height of the rectangular domain, = 'y_max' - 'y_min'
    'x_len': width of the rectangular domain, ='x_max' - 'x_min'
    'N': cell population at time t
    OUTPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates after migration
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and we append cell i's current velocity to it
    'vel_t': a 2D np array of N*2 storing instantaneous velocities of N cells 

    DEBUGGED
    '''

    # calculates the velocities then migrate;
    vel_t = (np.squeeze(M_rand_hat@F_rm[:,:,np.newaxis], axis=2) + np.squeeze(M_cc_hat@F_cc[:,:,np.newaxis],axis=2)) / eta

    # What if F_cc is in the direction of the v2 but lambda2=0? randomly choose from v1 and -v1 as a result of CG. 
    if len(zero_indices_cc) > 0:
        for i in range(len(zero_indices_cc)):
            # find v1 (thus -v1):
            omega_i = Omega_cell_loc[zero_indices_cc[i]]
            _, eigenvectors = np.linalg.eig(omega_i)
            v1 = eigenvectors[:, np.argmax(np.abs(eigenvectors[0]))] 
            v1 = v1 / np.linalg.norm(v1)
            v1_choices = np.array([v1, -v1])
            random_selectioni = v1_choices[np.random.choice([0, 1])]
            vel_i = (zero_CG_strengt_cc[i] * random_selectioni + \
                (1-zero_CG_strengt_cc[i]) * F_cc[zero_indices_cc[i]]) / eta
            vel_i = (vel_i/np.linalg.norm(vel_i)) * np.linalg.norm(F_cc[zero_indices_cc[i]]) # keep the magnitude of F_cc
            vel_t[zero_indices_cc[i]] += vel_i
    # similar for F_rm:
    if len(zero_indices_rm) > 0:
        for j in range(len(zero_indices_rm)):
            # find v1 (thus -v1):
            omega_j = Omega_cell_loc[zero_indices_rm[j]]
            _, eigenvectors = np.linalg.eig(omega_j)
            v1 = eigenvectors[:, np.argmax(np.abs(eigenvectors[0]))] 
            v1 = v1 / np.linalg.norm(v1)
            v1_choices = np.array([v1, -v1])
            random_selectionj = v1_choices[np.random.choice([0, 1])]
            vel_j = (zero_CG_strength_rm[j] * random_selectionj + \
                (1-zero_CG_strength_rm[j]) * F_rm[zero_indices_rm[j]]) / eta
            vel_j = (vel_j/np.linalg.norm(vel_j)) * np.linalg.norm(F_rm[zero_indices_rm[j]]) # keep the magnitude of F_cc
            vel_t[zero_indices_rm[j]] += vel_j

    # migrate cells based on velocities calculated:
    cell_coords = cell_coords + vel_t * delta_t

    # if periodic boundary conditions, update those cells falling outside the domain; otherwise, we don't remove those cells 
    # at this stage due to their later contributions to fibre dynamics. But after fibre dynamics, we will remove those cells.
    outside_x = np.logical_or(cell_coords[:, 0]<x_min, cell_coords[:, 0]>x_max) # in x direction
    if x_periodic_bool:
        cell_coords[outside_x, 0] = cell_coords[outside_x, 0] % x_len
    outside_y = np.logical_or(cell_coords[:, 1]<y_min, cell_coords[:, 1]>y_max) # in y-direction
    if y_periodic_bool:
        cell_coords[outside_y, 1] = cell_coords[outside_y, 1] % y_len

    # record velocity information for all cells in the domain into 'Vel_cells' for later fibre secretion purpose:
    for cell_i in range(N):
        Vel_cells[cell_i].append(vel_t[cell_i])

    return cell_coords, Vel_cells, vel_t

