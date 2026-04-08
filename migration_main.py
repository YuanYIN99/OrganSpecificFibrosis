# This script runs the single-cell, the cell-pair, and the cell-investigation showcases, and store the most detailed information into a pickle file.
import numpy as np
import random
import pickle
import os
import multiprocessing
import sys
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm 
import matplotlib.pyplot as plt
import itertools


from migration_subfunctions import *

def one_run(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
            omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, seed, store_path, rep, shift, scale, n, growth_rate, pool_top, cell_pool_width, \
                cellpool_bool, hstripe_N, topbool, bottombool, D_min):
    '''
    # This function runs simulation for the 'rep'th time based on the input parameter values, and store the time-lapsed cell and fibre dynamics into 'store_path/rep_{rep}.pickle'.
    '''

    store_name = store_path + f'rep_{rep}.pickle' # i.e. 'store_path/rep_{rep}.pickle'

    tstep = 60
    t_extract_times = list(np.arange(0, tend_iternum+tstep, tstep, dtype=int))
    #t_extract_times_minus1 = list(np.arange(tstep-1, tend_iternum-1+tstep, tstep, dtype=int))

    random.seed(seed)
    np.random.seed(seed)
    # note that we have checked in the local computer that seeding in Python affects the entire process, including all functions within that process. 
    # This means that if I seed the random number generator in a this function 'one_run', all subfunctions called within will also be affected by that seed.
    # HOWEVER, WE NEED TO CHECK WHETHER IT IS STILL THE CASE ON ARC. 

    # define lists to store cell coordinates, cell numbers, velocities, fibre field info throughout time, which will be output into a pickle file:
    cell_coords_T = [] 
    Omega_T = [] 
    cell_vel_NoCG_T = []
    cell_vel_T = [] 
    
    #cell_coords_T_minus1 = [] 
    #N_T = []
    #Prolif_coord_T = []
    # store the initial conditions:
    #cell_coords_T.append(cell_coords.copy())
    #Omega_T.append(Omega.copy())

    # initialise 'Vel_cells', which is a list of N empty, each will be used to record the history of a cell's (inside the domain) previous velocities.
    # this is useful for fibre secretion by cells. 
    N = int(len(cell_coords))
    #N_T.append(N)
    Vel_cells = [[] for _ in range(N)]

    # obtain inter-cellular forces experienced by all the cells:
    dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
    F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t)        

    # step throughout time by 1st order Euler:
    for t in range(tend_iternum+1):

        if t in t_extract_times:
            cell_coords_T.append(cell_coords.copy())
            Omega_T.append(Omega.copy())

        # proliferation if possible:
        # Note that proliferation shall be based on cells' positions before migration
        if prolif_bool:
            cell_coords_bf = cell_coords.copy()
            child_coords, mother_coord = cell_proliferation(Delta_0, rho_0, xi, prolif_bool, cell_coords_bf, num_rep_neigh, num_adh_neigh, N, y_periodic_bool, x_periodic_bool, \
                                                            y_len, x_len, x_min, x_max, y_min, y_max, delta_t, growth_rate) # note that cells are all within the domain, so no need to remove those outside the domain
            # update 'cell_coords' and 'Vel_cells' for those children cells:
            cell_coords = np.vstack((cell_coords, child_coords))
            Vel_cells = Vel_cells + [[] for _ in range(len(child_coords))]
            N = int(len(cell_coords))

        # extract fibre info at cells' locations: 
        Omega_cell_loc, total_fibre_cell_loc = fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N)
        # obtain random forces experienced by all the cells:
        F_rm = total_F_rm(D, N, delta_t, total_fibre_cell_loc, D_min) # SHALL WE PUT SEEDINGS IN ALL THE SUBFUNCTIONS? -- no HPC/ARC works fine
        # obtain inter-cellular forces experienced by all the cells:
        dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
        F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t)        
        # fibre contact guidance if possible:
        zero_indices_cc, zero_CG_strength_cc, zero_indices_rm, zero_CG_strength_rm = [], [], [], []
        if CG_bool:
            # evaluate fibre info at cell central locations:
            # contact guidance modulates random motion:
            M_rand_hat, zero_indices_rm, zero_CG_strength_rm = CG_rand(total_fibre_cell_loc, F_rm, Omega_cell_loc, N, shift, scale, n)
            # contact guidance modulates inter-cellular interactions:
            # OVER-COMPLICATED!
            #cf_dist, Omega_reshape, total_fibre_density, P_f = CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y)
            #Omega_cell_loc, total_fibre_cell_loc = fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N)
            #P_c = CGcc_Pc(F_adh_max, total_adh_magnitude, rep_dist_)
            #M_cc_hat = CG_cc(beta, F_cc, P_c, P_f, N, Omega_cell_loc)
            # SIMPLER VERSION:
            M_cc_hat, zero_indices_cc, zero_CG_strength_cc = CG_rand(total_fibre_cell_loc, F_cc, Omega_cell_loc, N, shift, scale, n)
            cf_dist, Omega_reshape, total_fibre_density = CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y)
        else:
            M_rand_hat = np.full((N, 2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]))
            M_cc_hat = np.full((N, 2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]))
        # migrate and update 'cell_coords' & 'Vel_cells':
        cell_coords_bf = cell_coords.copy() # this is used for later proliferation usage
        cell_coords, Vel_cells, vel_t = migrate_t(cell_coords, Vel_cells, M_cc_hat, M_rand_hat, F_rm, F_cc, eta, delta_t, y_periodic_bool, x_periodic_bool, x_min, x_max, x_len, y_min, y_max, y_len, N, 
                                                  zero_indices_cc, zero_CG_strength_cc, zero_indices_rm, zero_CG_strength_rm, Omega_cell_loc)
        # note that we are not removing rows of 'cell_coords', 'Vel_cells' and 'vel_t' when cells move outside the domain for their later usage in fibre dynamics. 
        if t in t_extract_times:
            cell_vel_T.append(vel_t.copy())
            vel_t_noCG = (F_rm + F_cc) / eta
            cell_vel_NoCG_T.append(vel_t_noCG.copy())

        # fibre dynamics driven by cells:
        # note that we use 'cf_dist', which is based on cell coordinates before migration. 
        # This makes sense, as we cells degrade and secrete fibres on grids that are under their impact areas BEFORE('_bf') migration.
        if CG_bool:
            Omega, Omega_reshape = fibre_degradation(Omega_reshape, cf_dist, sigma, d, omega_0, N, grid_x, grid_y, delta_t)
            Omega, Omega_reshape = fibre_secretion(Vel_cells, total_fibre_density, Omega_reshape, cf_dist, sigma, s, omega_0, N, grid_x, grid_y, delta_t, tau, num_rep_neigh)

        # now we are safe to remove cells that migrate outside the domain of interest:
        # in x direction:
        outside_x = np.logical_or(cell_coords[:, 0]<x_min, cell_coords[:, 0]>x_max)
        if ~x_periodic_bool:
            Vel_cells = [sublist for sublist, flag in zip(Vel_cells, outside_x) if not flag]
            cell_coords = cell_coords[~outside_x]
        # in y-direction:
        outside_y = np.logical_or(cell_coords[:, 1]<y_min, cell_coords[:, 1]>y_max)
        if ~y_periodic_bool:
            Vel_cells = [sublist for sublist, flag in zip(Vel_cells, outside_y) if not flag]
            cell_coords = cell_coords[~outside_y]

        # cell pool: confluency for the top and bottom cell pool areas:
        if cellpool_bool: 
            added_cellcoords, remove_cellindices = cellpool_confluency(pool_top, cell_coords, cell_pool_width, sigma, hstripe_N, y_max, x_min, x_max, y_min, topbool, bottombool)
            # remove if too many:
            if len(remove_cellindices) > 0:
                cell_coords = np.delete(cell_coords, remove_cellindices, axis=0)
                Vel_cells = [entry for i, entry in enumerate(Vel_cells) if i not in remove_cellindices]
            # add if too few
            if len(added_cellcoords) > 0:
                cell_coords = np.vstack((cell_coords, added_cellcoords))
                Vel_cells = Vel_cells + [[] for _ in range(len(added_cellcoords))]

        # update distribution of cells for later on proliferation:
        dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
        _, _, _, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t) 

        # get rid of numerical underflow problems with Omega
        mask_underflow = np.abs(Omega) <= 10**(-10) # YY: SHALL MATCH WITH RESULTS!!
        Omega[mask_underflow] = 0.0 

        N = int(len(cell_coords))

        #if t in t_extract_times:
        #    cell_coords_T.append(cell_coords.copy())
        #    N_T.append(N)
        #    Omega_T.append(Omega.copy())
        #    Prolif_coord_T.append(mother_coord.copy())
        #if t in t_extract_times_minus1: # storing cell coords for 'cell_vel_T' and 'cell_vel_NoCG_T'
        #    cell_coords_T_minus1.append(cell_coords.copy())

        # output into a pickle file 
        if t % 360.0 == 0.0:
            with open(store_name, 'wb') as file:
                pickle.dump(cell_coords_T, file)
                pickle.dump(Omega_T, file)
                pickle.dump(cell_vel_T, file)
                pickle.dump(cell_vel_NoCG_T, file)
                #pickle.dump(Prolif_coord_T, file)
                #pickle.dump(N_T, file)
                #pickle.dump(cell_coords_T_minus1, file)

    with open(store_name, 'wb') as file:
        pickle.dump(cell_coords_T, file)
        pickle.dump(Omega_T, file)
        pickle.dump(cell_vel_T, file)
        pickle.dump(cell_vel_NoCG_T, file)
        #pickle.dump(Prolif_coord_T, file)
        #pickle.dump(N_T, file)
        #pickle.dump(cell_coords_T_minus1, file)


    pass


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


def one_run_ConstSec(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
            omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, seed, store_path, rep, shift, scale, n, growth_rate, pool_top, cell_pool_width, \
                cellpool_bool, hstripe_N, topbool, bottombool, D_min):
    '''
    # This function runs simulation for the 'rep'th time based on the input parameter values, and store the time-lapsed cell and fibre dynamics into 'store_path/rep_{rep}.pickle'.
    '''

    store_name = store_path + f'rep_{rep}.pickle' # i.e. 'store_path/rep_{rep}.pickle'

    tstep = 60
    t_extract_times = list(np.arange(0, tend_iternum+tstep, tstep, dtype=int))
    #t_extract_times_minus1 = list(np.arange(tstep-1, tend_iternum-1+tstep, tstep, dtype=int))

    random.seed(seed)
    np.random.seed(seed)
    # note that we have checked in the local computer that seeding in Python affects the entire process, including all functions within that process. 
    # This means that if I seed the random number generator in a this function 'one_run', all subfunctions called within will also be affected by that seed.
    # HOWEVER, WE NEED TO CHECK WHETHER IT IS STILL THE CASE ON ARC. 

    # define lists to store cell coordinates, cell numbers, velocities, fibre field info throughout time, which will be output into a pickle file:
    cell_coords_T = [] 
    Omega_T = [] 
    cell_vel_NoCG_T = []
    cell_vel_T = [] 
    
    #cell_coords_T_minus1 = [] 
    #N_T = []
    #Prolif_coord_T = []
    # store the initial conditions:
    #cell_coords_T.append(cell_coords.copy())
    #Omega_T.append(Omega.copy())

    # initialise 'Vel_cells', which is a list of N empty, each will be used to record the history of a cell's (inside the domain) previous velocities.
    # this is useful for fibre secretion by cells. 
    N = int(len(cell_coords))
    #N_T.append(N)
    Vel_cells = [[] for _ in range(N)]

    # obtain inter-cellular forces experienced by all the cells:
    dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
    F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t)        

    # step throughout time by 1st order Euler:
    for t in range(tend_iternum+1):

        if t in t_extract_times:
            cell_coords_T.append(cell_coords.copy())
            Omega_T.append(Omega.copy())

        # proliferation if possible:
        # Note that proliferation shall be based on cells' positions before migration
        if prolif_bool:
            cell_coords_bf = cell_coords.copy()
            child_coords, mother_coord = cell_proliferation(Delta_0, rho_0, xi, prolif_bool, cell_coords_bf, num_rep_neigh, num_adh_neigh, N, y_periodic_bool, x_periodic_bool, \
                                                            y_len, x_len, x_min, x_max, y_min, y_max, delta_t, growth_rate) # note that cells are all within the domain, so no need to remove those outside the domain
            # update 'cell_coords' and 'Vel_cells' for those children cells:
            cell_coords = np.vstack((cell_coords, child_coords))
            Vel_cells = Vel_cells + [[] for _ in range(len(child_coords))]
            N = int(len(cell_coords))

        # extract fibre info at cells' locations: 
        Omega_cell_loc, total_fibre_cell_loc = fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N)
        # obtain random forces experienced by all the cells:
        F_rm = total_F_rm(D, N, delta_t, total_fibre_cell_loc, D_min) # SHALL WE PUT SEEDINGS IN ALL THE SUBFUNCTIONS? -- no HPC/ARC works fine
        # obtain inter-cellular forces experienced by all the cells:
        dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
        F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t)        
        # fibre contact guidance if possible:
        zero_indices_cc, zero_CG_strength_cc, zero_indices_rm, zero_CG_strength_rm = [], [], [], []
        if CG_bool:
            # evaluate fibre info at cell central locations:
            # contact guidance modulates random motion:
            M_rand_hat, zero_indices_rm, zero_CG_strength_rm = CG_rand(total_fibre_cell_loc, F_rm, Omega_cell_loc, N, shift, scale, n)
            # contact guidance modulates inter-cellular interactions:
            # OVER-COMPLICATED!
            #cf_dist, Omega_reshape, total_fibre_density, P_f = CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y)
            #Omega_cell_loc, total_fibre_cell_loc = fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N)
            #P_c = CGcc_Pc(F_adh_max, total_adh_magnitude, rep_dist_)
            #M_cc_hat = CG_cc(beta, F_cc, P_c, P_f, N, Omega_cell_loc)
            # SIMPLER VERSION:
            M_cc_hat, zero_indices_cc, zero_CG_strength_cc = CG_rand(total_fibre_cell_loc, F_cc, Omega_cell_loc, N, shift, scale, n)
            cf_dist, Omega_reshape, total_fibre_density = CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y)
        else:
            M_rand_hat = np.full((N, 2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]))
            M_cc_hat = np.full((N, 2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]))
        # migrate and update 'cell_coords' & 'Vel_cells':
        cell_coords_bf = cell_coords.copy() # this is used for later proliferation usage
        cell_coords, Vel_cells, vel_t = migrate_t(cell_coords, Vel_cells, M_cc_hat, M_rand_hat, F_rm, F_cc, eta, delta_t, y_periodic_bool, x_periodic_bool, x_min, x_max, x_len, y_min, y_max, y_len, N, 
                                                  zero_indices_cc, zero_CG_strength_cc, zero_indices_rm, zero_CG_strength_rm, Omega_cell_loc)
        # note that we are not removing rows of 'cell_coords', 'Vel_cells' and 'vel_t' when cells move outside the domain for their later usage in fibre dynamics. 
        if t in t_extract_times:
            cell_vel_T.append(vel_t.copy())
            vel_t_noCG = (F_rm + F_cc) / eta
            cell_vel_NoCG_T.append(vel_t_noCG.copy())

        # fibre dynamics driven by cells:
        # note that we use 'cf_dist', which is based on cell coordinates before migration. 
        # This makes sense, as we cells degrade and secrete fibres on grids that are under their impact areas BEFORE('_bf') migration.
        if CG_bool:
            Omega, Omega_reshape = fibre_degradation(Omega_reshape, cf_dist, sigma, d, omega_0, N, grid_x, grid_y, delta_t)
            Omega, Omega_reshape = fibre_secretion_CONST(Vel_cells, total_fibre_density, Omega_reshape, cf_dist, sigma, s, omega_0, N, grid_x, grid_y, delta_t, tau, num_rep_neigh)

        # now we are safe to remove cells that migrate outside the domain of interest:
        # in x direction:
        outside_x = np.logical_or(cell_coords[:, 0]<x_min, cell_coords[:, 0]>x_max)
        if ~x_periodic_bool:
            Vel_cells = [sublist for sublist, flag in zip(Vel_cells, outside_x) if not flag]
            cell_coords = cell_coords[~outside_x]
        # in y-direction:
        outside_y = np.logical_or(cell_coords[:, 1]<y_min, cell_coords[:, 1]>y_max)
        if ~y_periodic_bool:
            Vel_cells = [sublist for sublist, flag in zip(Vel_cells, outside_y) if not flag]
            cell_coords = cell_coords[~outside_y]

        # cell pool: confluency for the top and bottom cell pool areas:
        if cellpool_bool: 
            added_cellcoords, remove_cellindices = cellpool_confluency(pool_top, cell_coords, cell_pool_width, sigma, hstripe_N, y_max, x_min, x_max, y_min, topbool, bottombool)
            # remove if too many:
            if len(remove_cellindices) > 0:
                cell_coords = np.delete(cell_coords, remove_cellindices, axis=0)
                Vel_cells = [entry for i, entry in enumerate(Vel_cells) if i not in remove_cellindices]
            # add if too few
            if len(added_cellcoords) > 0:
                cell_coords = np.vstack((cell_coords, added_cellcoords))
                Vel_cells = Vel_cells + [[] for _ in range(len(added_cellcoords))]

        # update distribution of cells for later on proliferation:
        dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
        _, _, _, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t) 

        # get rid of numerical underflow problems with Omega
        mask_underflow = np.abs(Omega) <= 10**(-10) # YY: SHALL MATCH WITH RESULTS!!
        Omega[mask_underflow] = 0.0 

        N = int(len(cell_coords))

        #if t in t_extract_times:
        #    cell_coords_T.append(cell_coords.copy())
        #    N_T.append(N)
        #    Omega_T.append(Omega.copy())
        #    Prolif_coord_T.append(mother_coord.copy())
        #if t in t_extract_times_minus1: # storing cell coords for 'cell_vel_T' and 'cell_vel_NoCG_T'
        #    cell_coords_T_minus1.append(cell_coords.copy())

        # output into a pickle file 
        if t % 360.0 == 0.0:
            with open(store_name, 'wb') as file:
                pickle.dump(cell_coords_T, file)
                pickle.dump(Omega_T, file)
                pickle.dump(cell_vel_T, file)
                pickle.dump(cell_vel_NoCG_T, file)
                #pickle.dump(Prolif_coord_T, file)
                #pickle.dump(N_T, file)
                #pickle.dump(cell_coords_T_minus1, file)

    with open(store_name, 'wb') as file:
        pickle.dump(cell_coords_T, file)
        pickle.dump(Omega_T, file)
        pickle.dump(cell_vel_T, file)
        pickle.dump(cell_vel_NoCG_T, file)
        #pickle.dump(Prolif_coord_T, file)
        #pickle.dump(N_T, file)
        #pickle.dump(cell_coords_T_minus1, file)


    pass



def scratchassay(opt, numCPUs):

    if opt == 1: # proliferation assay (cell-fibre): CF with a bigger radius: for tp = 25
        # parameters:
        delta_t, tend_iternum, Numrep = 1.0, 5040, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1740.0, 0.0, 1290.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # random motion:
        D = 0.01 * (1.9**10) 
        D_min = D / 10.0
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 71.0*2, 1.0 
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        epsilon = 0.01 * (1.4**0) 
        F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool = True
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool (for scratch assay): top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        # THIS IS NOT USED
        cell_pool_width = 300.0 # this divides sigma = 50.0 
        pool_top = y_max-cell_pool_width
        hstripe_N = int(33) 
        cellpool_bool, topbool, bottombool = False, False, False

        # init cell coordinates
        cell_coords_init = np.load('ICs/proliferation_CF_cells_IC.npy')
        # init collagen fibre field (with fibre model: void of fibres)
        Omega_init = np.load('ICs/proliferation_CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/proliferation_CF_fibre_coords.npy')
        grid_x = np.load('ICs/proliferation_CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/proliferation_CF_fibre_grid_y.npy')

        FC = np.array([1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0])
        parameters = 4320.0/np.log(FC) # Delta_0s
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            Delta_0 = parameters[j]
            sub_path = f'Proliferation_/CF_25tp/sigma{sigma}_D10_epsilon1_FC{FC[j]}_NS0.2/'

            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 11: # proliferation assay (cell-fibre): CF with a bigger radius: for tp = 32
        # parameters:
        delta_t, tend_iternum, Numrep = 1.0, 5040, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1740.0, 0.0, 1290.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # random motion:
        D = 0.01 * (1.9**9) 
        D_min = D / 10.0
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 71.0*2, 1.0 
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        epsilon = 0.01 * (1.4**2) 
        F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool = True
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool (for scratch assay): top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        # THIS IS NOT USED
        cell_pool_width = 300.0 # this divides sigma = 50.0 
        pool_top = y_max-cell_pool_width
        hstripe_N = int(33) 
        cellpool_bool, topbool, bottombool = False, False, False

        # init cell coordinates
        cell_coords_init = np.load('ICs/proliferation_CF_cells_IC.npy')
        # init collagen fibre field (with fibre model: void of fibres)
        Omega_init = np.load('ICs/proliferation_CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/proliferation_CF_fibre_coords.npy')
        grid_x = np.load('ICs/proliferation_CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/proliferation_CF_fibre_grid_y.npy')

        FC = np.array([1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0])
        parameters = 4320.0/np.log(FC) # Delta_0s
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            Delta_0 = parameters[j]
            sub_path = f'Proliferation_/CF_32tp/sigma{sigma}_D10_epsilon1_FC{FC[j]}_NS0.2/'

            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 2: # proliferation assay (cell-fibre): LF with a smaller radius
        # parameters:
        delta_t, tend_iternum, Numrep = 1.0, 5040, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1740.0, 0.0, 1290.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # random motion:
        D = 0.01 * (1.9**10) 
        D_min = D / 10.0
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 41.0*2, 1.0 
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        epsilon = 0.01 * (1.4**1) 
        F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool = True
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool (for scratch assay): top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        # THIS IS NOT USED
        cell_pool_width = 300.0 # this divides sigma = 50.0 
        pool_top = y_max-cell_pool_width
        hstripe_N = int(33) 
        cellpool_bool, topbool, bottombool = False, False, False

        # init cell coordinates
        cell_coords_init = np.load('ICs/proliferation_LF_cells_IC.npy')
        # init collagen fibre field (with fibre model: void of fibres)
        Omega_init = np.load('ICs/proliferation_LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/proliferation_LF_fibre_coords.npy')
        grid_x = np.load('ICs/proliferation_LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/proliferation_LF_fibre_grid_y.npy')

        #FC = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0])
        FC = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0])
        parameters = 4320.0/np.log(FC) # Delta_0s
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            Delta_0 = parameters[j]
            sub_path = f'Proliferation_/LF/sigma{sigma}_D10_epsilon1_FC{FC[j]}_NS0.2/'
            
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)
    
    if opt == 3: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 0-5
        parameters, parameter_names = parameters[0:5], parameter_names[0:5]


        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 33: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 5-10
        parameters, parameter_names = parameters[5:10], parameter_names[5:10]


        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 10-15
        parameters, parameter_names = parameters[10:15], parameter_names[10:15]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 3333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 15-20
        parameters, parameter_names = parameters[15:20], parameter_names[15:20]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 33333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 20-25
        parameters, parameter_names = parameters[20:25], parameter_names[20:25]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 25-30
        parameters, parameter_names = parameters[25:30], parameter_names[25:30]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 3333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 30:35
        parameters, parameter_names = parameters[30:35], parameter_names[30:35]


        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 33333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 35:40
        parameters, parameter_names = parameters[35:40], parameter_names[35:40]


        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 333333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 40:45
        parameters, parameter_names = parameters[40:45], parameter_names[40:45]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 3333333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 45:50
        parameters, parameter_names = parameters[45:50], parameter_names[45:50]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 33333333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 50:55
        parameters, parameter_names = parameters[50:55], parameter_names[50:55]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 333333333333: # scratch assay: CF with a smaller radius
        # parameters:
        cell_poollen = 50.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 46.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/CF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/CF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/CF_fibre_coords.npy')
        grid_x = np.load('ICs/CF_fibre_grid_x.npy')
        grid_y = np.load('ICs/CF_fibre_grid_y.npy')

        #exponent_D = np.arange(0, 13, 1)
        #Ds = 0.01 * 1.90 ** (exponent_D)
        #exponent_e = np.arange(0, 13, 1)
        #epsilons = 0.01 * 1.4 ** (exponent_e)
        Ds = np.arange(1.0, 13.0, 1.0)
        epsilons = np.arange(0.01, 0.06, 0.01)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        #parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(Ds, epsilons)]
        # submitted: 55:
        parameters, parameter_names = parameters[55:], parameter_names[55:]

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_CF_FINER/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)



    if opt == 4: # scratch assay: LF with a bigger radius
        # parameters:
        cell_poollen = 90.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 83.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/LF_fibre_coords.npy')
        grid_x = np.load('ICs/LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/LF_fibre_grid_y.npy')

        exponent_D = np.arange(0, 13, 1)
        Ds = 0.01 * 1.90 ** (exponent_D)
        exponent_e = np.arange(0, 13, 1)
        epsilons = 0.01 * 1.4 ** (exponent_e)
        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        
        # submitted: 40-60
        # running: 0-20
        parameters, parameter_names = parameters[40:60], parameter_names[40:60]
        
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_LF_NEW/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 44: # scratch assay: LF with a bigger radius
        # parameters:
        cell_poollen = 90.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 83.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/LF_fibre_coords.npy')
        grid_x = np.load('ICs/LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/LF_fibre_grid_y.npy')

        exponent_D = np.arange(0, 13, 1)
        Ds = 0.01 * 1.90 ** (exponent_D)
        exponent_e = np.arange(0, 13, 1)
        epsilons = 0.01 * 1.4 ** (exponent_e)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        # running: 20-40
        # submitted: 80-100
        parameters, parameter_names = parameters[80:100], parameter_names[80:100]
        
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_LF_NEW/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 444: # scratch assay: LF with a bigger radius
        # parameters:
        cell_poollen = 90.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 83.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/LF_fibre_coords.npy')
        grid_x = np.load('ICs/LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/LF_fibre_grid_y.npy')

        exponent_D = np.arange(0, 13, 1)
        Ds = 0.01 * 1.90 ** (exponent_D)
        exponent_e = np.arange(0, 13, 1)
        epsilons = 0.01 * 1.4 ** (exponent_e)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        # submitted: 60-80
        parameters, parameter_names = parameters[60:80], parameter_names[60:80]
        
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_LF_NEW/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 4444: # scratch assay: LF with a bigger radius
        # parameters:
        cell_poollen = 90.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 83.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/LF_fibre_coords.npy')
        grid_x = np.load('ICs/LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/LF_fibre_grid_y.npy')

        exponent_D = np.arange(0, 13, 1)
        Ds = 0.01 * 1.90 ** (exponent_D)
        exponent_e = np.arange(0, 13, 1)
        epsilons = 0.01 * 1.4 ** (exponent_e)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        # submitted: 100-120
        parameters, parameter_names = parameters[100:120], parameter_names[100:120]
        
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_LF_NEW/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 44444: # scratch assay: LF with a bigger radius
        # parameters:
        cell_poollen = 90.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 83.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/LF_fibre_coords.npy')
        grid_x = np.load('ICs/LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/LF_fibre_grid_y.npy')

        exponent_D = np.arange(0, 13, 1)
        Ds = 0.01 * 1.90 ** (exponent_D)
        exponent_e = np.arange(0, 13, 1)
        epsilons = 0.01 * 1.4 ** (exponent_e)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        # submitted: 120-140
        parameters, parameter_names = parameters[120:140], parameter_names[120:140]
        
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_LF_NEW/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 444444: # scratch assay: LF with a bigger radius
        # parameters:
        cell_poollen = 90.0 # this shall be bigger than sigma and divides 10.0 (fibre grid size)
        delta_t, tend_iternum, Numrep = 1.0, 4320, 10 
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1200.0+3*cell_poollen, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 83.0, 1.0
        r_max, rep_adh_len = 3*sigma, (2**(1/6))*sigma
        # proliferation:
        prolif_bool, Delta_0 = True, 4320.0/np.log(1.5)
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG: (This is a cell-only model)
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d, tau = 1.0, 0.0005, 0.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, True, True

        # init cell coordinates
        cellinit_filename = 'ICs/LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/LF_fibre_coords.npy')
        grid_x = np.load('ICs/LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/LF_fibre_grid_y.npy')

        exponent_D = np.arange(0, 13, 1)
        Ds = 0.01 * 1.90 ** (exponent_D)
        exponent_e = np.arange(0, 13, 1)
        epsilons = 0.01 * 1.4 ** (exponent_e)

        parameters = np.array([np.array([d, e]) for d, e in itertools.product(Ds, epsilons)]) # =[D, epsilon]
        parameter_names = [f'D{d}_epsilon_{e}' for d, e in itertools.product(exponent_D, exponent_e)]
        # submitted: 140-169
        parameters, parameter_names = parameters[140:], parameter_names[140:]
        
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 

        for j in range(len(parameters)):
            D, epsilon = parameters[j, 0], parameters[j, 1]
            D_min = D / 10.0
            F_adh_max = 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))

            sub_path = f'Scratch_LF_NEW/sigma{sigma}_' + parameter_names[j] + '_NS0.2/'
            print(sub_path)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    if opt == 5: # One-sided larger scratch: memory length
        # parameters:
        delta_t, tend_iternum, Numrep = 1.0, 8640, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, -150.0, 1600.0, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        D = 0.01 * (1.9**10)
        D_min = D / 10.0
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 48.0, 1.0
        epsilon = 0.01 * (1.4**1)
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool, Delta_0 = True, 6000.0
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG:
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, s, d = 1.0, 0.0005, 0.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = 150.0 # this divides sigma = 50.0
        pool_top = y_max-cell_pool_width
        hstripe_N = int(43) 
        cellpool_bool, topbool, bottombool = True, False, True

        # init cell coordinates
        cellinit_filename = 'ICs/OnesidedScratch_cell.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)
        # init collagen fibre field
        Omega_init = np.load('ICs/OnesidedScratch_Omega.npy')
        fibre_coords = np.load('ICs/OnesidedScratch_fibre_coords.npy')
        grid_x = np.load('ICs/OnesidedScratch_grid_x.npy')
        grid_y = np.load('ICs/OnesidedScratch_grid_y.npy')

        parameters = np.array([300.0, 600.0, 900, 1200.0])
        parameters = np.array([1200.0])
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep))

        for j in range(len(parameters)):
            tau = parameters[j]

            sub_path = f'OnesidedScratch/sigma{sigma}_D10_epsilon1_memorylen{tau}_NS0.2/'
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)
    
    if opt == 6: # One-sided larger scratch: secretion and degradation rates for LF
        # parameters:
        cell_poollen = 90.0 
        delta_t, tend_iternum, Numrep = 1.0, 8640, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1600.0, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        D = 0.01 * (1.9**10) 
        D_min = D / 10.0
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 89.0, 1.0
        epsilon = 0.01 * (1.4**1)
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool, Delta_0 = True, 6000.0
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG:
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, tau = 1.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, False, True

        # init cell coordinates
        cellinit_filename = 'ICs/OneSided_LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/OneSided_LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/OneSided_LF_fibre_coords.npy')
        grid_x = np.load('ICs/OneSided_LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/OneSided_LF_fibre_grid_y.npy')

        #s_and_d = np.array([0.0005, 0.0005/2, 0.0005*2])
        #parameters = np.array([np.array([s, d]) for s, d in itertools.product(s_and_d, s_and_d)])
        parameters = np.array([[0.0005, 0.0], 
                               [0.005, 0.0], 
                               [0.05, 0.0], 
                               [0.5, 0.0], 
                               [0.5, 0.0005], 
                               [0.5, 0.005], 
                               [0.5, 0.05], 
                               [0.5, 0.5]])

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep))

        for j in range(len(parameters)):
            s, d = parameters[j, 0], parameters[j, 1]

            sub_path = f'OnesidedScratch_/SecDeg/sigma{sigma}_D10_epsilon1_s{s}_d{d}_NS0.2/'
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 7: # One-sided larger scratch: constant D for LF
        # parameters:
        cell_poollen = 90.0 
        delta_t, tend_iternum, Numrep = 1.0, 8640, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1600.0, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        D = 0.01 * (1.9**10) 
        D_min = D
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 89.0, 1.0
        epsilon = 0.01 * (1.4**1)
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool, Delta_0 = True, 6000.0
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG:
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, tau = 1.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, False, True

        # init cell coordinates
        cellinit_filename = 'ICs/OneSided_LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/OneSided_LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/OneSided_LF_fibre_coords.npy')
        grid_x = np.load('ICs/OneSided_LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/OneSided_LF_fibre_grid_y.npy')

        #s_and_d = np.array([0.0005, 0.0005/2, 0.0005*2])
        #parameters = np.array([np.array([s, d]) for s, d in itertools.product(s_and_d, s_and_d)])
        parameters = np.array([
                               [0.5, 0.0], 
                               [0.05, 0.0], 
                               [0.5, 0.0005], 
                               [0.5, 0.005], 
                               [0.5, 0.05], 
                               [0.5, 0.5], 
                               [0.005, 0.0], 
                               [0.0005, 0.0], ])

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep))

        for j in range(len(parameters)):
            s, d = parameters[j, 0], parameters[j, 1]

            sub_path = f'OnesidedScratch_/ConstD_SecDeg/sigma{sigma}_D10_epsilon1_s{s}_d{d}_NS0.2/'
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    if opt == 8: # One-sided larger scratch: constant Sec for LF
        # parameters:
        cell_poollen = 90.0 
        delta_t, tend_iternum, Numrep = 1.0, 8640, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1600.0, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        D = 0.01 * (1.9**10) 
        D_min = D / 10.0
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 89.0, 1.0
        epsilon = 0.01 * (1.4**1)
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool, Delta_0 = True, 6000.0
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG:
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, tau = 1.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, False, True

        # init cell coordinates
        cellinit_filename = 'ICs/OneSided_LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/OneSided_LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/OneSided_LF_fibre_coords.npy')
        grid_x = np.load('ICs/OneSided_LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/OneSided_LF_fibre_grid_y.npy')

        #s_and_d = np.array([0.0005, 0.0005/2, 0.0005*2])
        #parameters = np.array([np.array([s, d]) for s, d in itertools.product(s_and_d, s_and_d)])
        parameters = np.array([
                               [0.5, 0.0], 
                               [0.05, 0.0], 
                               [0.5, 0.0005], 
                               [0.5, 0.005], 
                               [0.5, 0.05], 
                               [0.5, 0.5], 
                               [0.005, 0.0], 
                               [0.0005, 0.0], ])

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep))

        for j in range(len(parameters)):
            s, d = parameters[j, 0], parameters[j, 1]

            sub_path = f'OnesidedScratch_/ConstSec_SecDeg/sigma{sigma}_D10_epsilon1_s{s}_d{d}_NS0.2/'
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run_ConstSec, arguments)

    if opt == 9: # One-sided larger scratch: constant Sec and constant D for LF
        # parameters:
        cell_poollen = 90.0 
        delta_t, tend_iternum, Numrep = 1.0, 8640, 10
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 1600.0, 0.0-3*cell_poollen, 1600.0, False, True
        x_len, y_len = x_max-x_min, y_max-y_min
        D = 0.01 * (1.9**10) 
        D_min = D
        # cell-cell interaction: 
        eta, sigma, beta = 1.0, 89.0, 1.0
        epsilon = 0.01 * (1.4**1)
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        # proliferation:
        prolif_bool, Delta_0 = True, 6000.0
        rho_0 = 6.0
        xi = sigma/2
        growth_rate = 0.0001 # NOT USED (THIS IS FOR SIMPLE EXP GROWTH MODEL)
        # CG:
        CG_bool = True
        scale, n = 10.0, 1
        shift = 0.2
        omega_0, tau = 1.0, 300.0
        
        # cell pool: top pool: pool_top<=y<=y_max, bottom pool: y_min<=y<=pool_bottom
        cell_pool_width = cell_poollen * 3
        pool_top = y_max-cell_pool_width
        hstripe_N = int(sigma*x_max/(np.pi * (sigma/2)**2))
        cellpool_bool, topbool, bottombool = True, False, True

        # init cell coordinates
        cellinit_filename = 'ICs/OneSided_LF_cells.npy'
        cell_coords_init = np.load(cellinit_filename)
        # init collagen fibre field
        Omega_init = np.load('ICs/OneSided_LF_fibre_Omega.npy')
        fibre_coords = np.load('ICs/OneSided_LF_fibre_coords.npy')
        grid_x = np.load('ICs/OneSided_LF_fibre_grid_x.npy')
        grid_y = np.load('ICs/OneSided_LF_fibre_grid_y.npy')

        #s_and_d = np.array([0.0005, 0.0005/2, 0.0005*2])
        #parameters = np.array([np.array([s, d]) for s, d in itertools.product(s_and_d, s_and_d)])
        parameters = np.array([
                               [0.5, 0.0], 
                               [0.05, 0.0], 
                               [0.5, 0.0005], 
                               [0.5, 0.005], 
                               [0.5, 0.05], 
                               [0.5, 0.5], 
                               [0.005, 0.0], 
                               [0.0005, 0.0], ])

        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep))

        for j in range(len(parameters)):
            s, d = parameters[j, 0], parameters[j, 1]

            sub_path = f'OnesidedScratch_/ConstSecD_SecDeg/sigma{sigma}_D10_epsilon1_s{s}_d{d}_NS0.2/'
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path, rep, shift, scale, n, growth_rate, pool_top, \
                            cell_pool_width, cellpool_bool, hstripe_N, topbool, bottombool, D_min) for rep in range(Numrep)]
                pool.starmap(one_run_ConstSec, arguments)

    
    
    
    
    
    pass


if __name__ == "__main__":

    opt = int(sys.argv[1])  
    numCPUs = int(sys.argv[2]) 
    scratchassay(opt, numCPUs)


