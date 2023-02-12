"""
1D MCMC inversion

:Copyright:
    Author: Chuanming Liu
    Department of Physics, University of Colorado Boulder
"""

import numpy as np
from os.path import exists, join
import os 
import multiprocessing as mp 
from functools import partial 
import copy 
import pdb 
import logging.config
import random
import time
import glob
logger = logging.getLogger(__name__)

# local 
from .data import Data
from .vmodel import Model1d
from .forward import Forward

# priviate
class MisFit(Data):
    """
    Class for data used invesion 
    self.hti_azi_only: True: only use Azi data to calculate the misfit
    """
    def __init__(self, modtype='tti', hti_azi_only=False):
        super().__init__()
        if modtype == 'tti':
            self.get_misfit = self.get_misfit_tti
        elif modtype == 'hti':
            self.get_misfit = self.get_misfit_hti
        elif modtype == 'iso':
            self.get_misfit = self.get_misfit_iso
        self.hti_azi_only = hti_azi_only
        return

    def adapt_data(self, data):
        self.dispR = data.dispR
        self.dispL = data.dispL 
        self.dispPhi = data.dispPhi 
        self.dispAmp = data.dispAmp 

    def get_prediction(self, dataPred):
        
        if self.dispR.nper >0: 
            self.dispR.valsPred = dataPred.dispR.vals

        if self.dispL.nper >0:
            self.dispL.valsPred = dataPred.dispL.vals

        if self.dispPhi.nper > 0:
            self.dispPhi.valsPred = dataPred.dispPhi.vals

        if self.dispAmp.nper > 0:
            self.dispAmp.valsPred = dataPred.dispAmp.vals     
        return
    
    def get_misfit_tti(self):
        """
        Joint chi-squre and misift

        Chi-squre: S = sum((pred-obs)**2/(sigma)**2)
        Joint: S = S_dispR + S_dispL + S_dispAmp + S_dispPhi

        Joint misfit (reduced chi-square misfit): sqrt(S/N_total)

        Likelihood: exp(-0.5*S)
        """

        chiSqr_dispR, _, _ = self.dispR.get_misfit()
        chiSqr_dispL, _, _ = self.dispL.get_misfit()
        chiSqr_dispPhi, _, _ = self.dispPhi.get_misfit()
        chiSqr_dispAmp, _, _ = self.dispAmp.get_misfit()


        chiSqr = chiSqr_dispR + chiSqr_dispL + chiSqr_dispPhi + chiSqr_dispAmp 
        Nt = self.dispR.nper + self.dispL.nper + self.dispPhi.nper + self.dispAmp.nper
        misfit = np.sqrt(chiSqr/Nt)

        # !!!!! # likelihood 
        chiSqr, L = self._reduced_likelihood(chiSqr)

        # pre-judge
        chiSqr_pre = chiSqr_dispR + chiSqr_dispL + chiSqr_dispPhi
        Nt_pre = self.dispR.nper + self.dispL.nper + self.dispPhi.nper 
        misfit_pre = np.sqrt(chiSqr_pre/Nt_pre)

        # !!!!!!
        # old_version 
        chiSqr_pre, L_pre = self._reduced_likelihood(chiSqr_pre)

        self.misfit = misfit 
        self.L = L 
        self.misfit_pre = misfit_pre 
        self.L_pre = L_pre

        return L, L_pre, chiSqr, chiSqr_pre, misfit

    def _reduced_likelihood(self, chiSqr):
        # old_version # likelihood 
        chiSqr = 0.5 * chiSqr
        for i in range(4):
            chiSqr =  chiSqr if chiSqr < 50 else np.sqrt(chiSqr*50.) 
        L = np.exp(-0.5 * chiSqr)
        return chiSqr, L

    def get_misfit_hti(self):
        """
        Joint chi-squre and misift
        HTI: fit only C_R, amp_R, phi_R

        Chi-squre: S = sum((pred-obs)**2/(sigma)**2)
        Joint: S = S_dispR + S_dispAmp + S_dispPhi

        Joint misfit (reduced chi-square misfit): sqrt(S/N_total)

        Likelihood: exp(-0.5*S)
        """
        if not self.hti_azi_only:
            chiSqr_dispR, _, _ = self.dispR.get_misfit()
        chiSqr_dispPhi, _, _ = self.dispPhi.get_misfit()   
        chiSqr_dispAmp, _, _ = self.dispAmp.get_misfit()

        if self.hti_azi_only:
            chiSqr = chiSqr_dispPhi + chiSqr_dispAmp 
            Nt = self.dispPhi.nper + self.dispAmp.nper
        else:
            chiSqr = chiSqr_dispR + chiSqr_dispPhi + chiSqr_dispAmp 
            Nt = self.dispR.nper + self.dispPhi.nper + self.dispAmp.nper
        misfit = np.sqrt(chiSqr/Nt)

        # !!!!! # likelihood 
        chiSqr, L = self._reduced_likelihood(chiSqr)

        # !!!!!!
        # old_version 
        # chiSqr_pre, L_pre = self._reduced_likelihood(chiSqr_pre)

        self.misfit = misfit 
        self.L = L 
        self.misfit_pre = misfit 
        self.L_pre = L

        return L, None, chiSqr, None, misfit

    def get_misfit_iso(self):
        """
        Joint chi-squre and misift
        HTI: fit only C_R, amp_R, phi_R

        Chi-squre: S = sum((pred-obs)**2/(sigma)**2)
        Joint: S = S_dispR + S_dispAmp + S_dispPhi

        Joint misfit (reduced chi-square misfit): sqrt(S/N_total)

        Likelihood: exp(-0.5*S)
        """

        if self.dispR.nper >0: 
            chiSqr_dispR, _, _ = self.dispR.get_misfit()

        chiSqr = chiSqr_dispR 
        Nt = self.dispR.nper 
        misfit = np.sqrt(chiSqr/Nt)
        # !!!!! # likelihood 
        chiSqr, L = self._reduced_likelihood(chiSqr)

        self.misfit = misfit 
        self.L = L 
        self.misfit_pre = misfit 
        self.L_pre = L

        return L, None, chiSqr, None, misfit


    def cal_misfit(self, dataPred):
        self.get_prediction(dataPred)
        return self.get_misfit()        

    def copy(self):
        return copy.deepcopy(self)


class McTrack(object):
    
    def __init__(self, npara=None, Nrun=None, data=None):
        self.npara = npara
        self.nper_R, self.nper_L, self.nper_Phi = [data.dispR.nper, data.dispL.nper, data.dispPhi.nper]

        self.outmodarr = np.zeros((Nrun, npara+12))   # original
        if self.nper_R > 0:
            self.out_ray = np.zeros((Nrun, data.dispR.nper))
        if self.nper_L > 0:
            self.out_lov = np.zeros((Nrun, data.dispL.nper))
        if self.nper_Phi > 0:
            self.out_azi = np.zeros((Nrun, data.dispPhi.nper))
            self.out_amp = np.zeros((Nrun, data.dispPhi.nper))

    def save_model(self, model, ind, mark_acc, ind_acc, Misfit=None, markKernel=None):
        """
        ind            - index of interation
        model          - Model1d()
        data           - Data()
        mark_acc       - index for acceptance 
                        (1: accept; -1: rejected; -2: pre-rejected; [-3~-5]: problematic model for recalculation of dcdL)
        ind_acc        - index of accepted models 
        likeliMisfit   - [data.L, data.misfit, data.L_pre, data.misfit_pre]
        dataMisfit     - MisFit() current
        markKernel     - signal of update of model; 
                        0: no kernel update; 1: large dcdL updated/ uniform random ; 
                        2: forward kernel updated for Rayleigh  3: forward kernel updated for Love
                        4: forward kernel updated for both
        """
        npara = self.npara
        self.outmodarr[ind, 0] = mark_acc             
        self.outmodarr[ind, 1] = ind_acc             
        self.outmodarr[ind, 2:(npara+2)] = model.ttimod.para.paraval

        # 4 likeli + misfit
        self.outmodarr[ind, npara+2 : npara+6] = [Misfit.L, Misfit.misfit, Misfit.L_pre, Misfit.misfit_pre]

        # data misfit
        self.outmodarr[ind, npara+6: npara+10] = [Misfit.dispR.misfit, Misfit.dispL.misfit, Misfit.dispPhi.misfit, Misfit.dispAmp.misfit]
        # if self.rf_MC: outmodarr[ind, npara+10]  = self.data.rfr.misfit

        if markKernel is not None:
            self.outmodarr[ind, npara+11] = markKernel        
        return 
    
    def save_data(self, Misfit, ind):
        """ 
        """
        if self.nper_R > 0:
            self.out_ray[ind, :] = Misfit.dispR.valsPred[:]
        if self.nper_L > 0:
            self.out_lov[ind, :] = Misfit.dispL.valsPred[:]
        if self.nper_Phi > 0:
            self.out_azi[ind, :] = Misfit.dispPhi.valsPred[:]
            self.out_amp[ind, :] = Misfit.dispAmp.valsPred[:]
        return

    def output_MC(self, fnm, outdir='.'):
        outfnm = join(outdir, fnm)
        if self.nper_R * self.nper_L * self.nper_Phi > 0:
            np.savez_compressed(outfnm, outmodarr=self.outmodarr, out_ray=self.out_ray, out_lov=self.out_lov, out_azi=self.out_azi, out_amp=self.out_amp)
        elif self.nper_R * self.nper_L > 0:
            np.savez_compressed(outfnm, outmodarr=self.outmodarr, out_ray=self.out_ray, out_lov=self.out_lov)
        elif self.nper_R * self.nper_Phi > 0:
            np.savez_compressed(outfnm, outmodarr=self.outmodarr, out_ray=self.out_ray, out_azi=self.out_azi, out_amp=self.out_amp)
        elif self.nper_Phi > 0:
            np.savez_compressed(outfnm, outmodarr=self.outmodarr, out_azi=self.out_azi, out_amp=self.out_amp)
        elif self.nper_R > 0:
            np.savez_compressed(outfnm, outmodarr=self.outmodarr, out_ray=self.out_ray)
        return 

class Point(object):
    """ 
    Object for 1D depth-dependent elastic module profile, inversion process
    ======================
    :: parameters ::

    data   - object of input data 
    model  - object of 1D model

    McTrack - t
    ======================
    """
    def __init__(self, modtype='tti', inv_thickness=False, hti_azi_only=False): 
        self.dataObs = Data() 
        self.dataPred = Data()
        self.initMod = Model1d(mod=modtype, inv_thickness=inv_thickness)
        self.Misfit = MisFit(modtype=modtype, hti_azi_only=hti_azi_only)
        self.pid = 'test'
        self.modtype = modtype
        return
    
    def readObsData(self, dtype='Ray_ph', infnm='', **kwargs):
        """
        load observations from input
        """
        self.dataObs.readInput(dtype=dtype, infnm=infnm, **kwargs)
        return 

    def init_misfit_forward(self, **kwargs):
        self.Misfit.adapt_data(self.dataObs)
        self.forward = Forward(dataObs=self.dataObs, modtype=self.modtype, **kwargs)
        return
        
    def init_forward(self):
        """
        For posterior 
        """
        self.forward = Forward(dataObs=self.dataObs, modtype=self.modtype)
        return 

    def readmod(self, modfnm='', elltype=True, region='Alaska'):
        """
        load model file 
        """
        self.initMod.ttimod.readtimodtxt(infname=modfnm, ellipytical=elltype, region=region)

        return 

    def accept(self, chiSqr0, chiSqr1):
        if chiSqr1 < chiSqr0:
            return True 
        else:
            return random.random() > 1 - np.exp(-(chiSqr1-chiSqr0)/2) # (L0-L1)/L0

    def accept_likeli(self, likeli_old, likeli_new):
        if likeli_new > likeli_old: 
            return True 
        else:
            prob = (likeli_old - likeli_new)/likeli_old
            return random.random() > prob

    def forward2misfit(self, model=None, wtype='both', inv_type='tti', solver_str='normal', refc_str='fsurf', **kwargs):
        """ 
        solver_str - 'normal': direct calculate disp; 'perturbation': based on the kernel saved in self.forward
        wtype      - 'both'; Note: when wtype is 'Ray' or 'Lov', the another type data will use the save item in self.forward.dataPred
        """
        if model is None:
            model = self.initMod 

        # forward 
        data_pred = self.forward.CalSurf(model=model, inv_type=inv_type, wtype=wtype, solver_str=solver_str, refc_str=refc_str, **kwargs)

        self.dataPred = data_pred


        # misfit
        # self.Misfit.get_prediction(data_pred)
        # L, L_pre, chiSqr, chiSqr_pre, misfit  = self.Misfit.get_misfit()
        L, L_pre, chiSqr, chiSqr_pre, misfit  = self.Misfit.cal_misfit(data_pred)

        return L, L_pre, chiSqr, chiSqr_pre, misfit

    def reset_forward(self, model=None, wtypes='both', inv_type=None, refc_str='fsurf'):
        """ 
        Re-calculate misfit for large perturbation in disp
        """
        if inv_type is None:
            inv_type = self.modtype
        if 'ray' in wtypes:
            data_pred = self.forward.CalSurf(model=model, inv_type=inv_type, wtype='ray', solver_str='normal', refc_str=refc_str)
            irefmd = 2

        if 'lov' in wtypes:
            data_pred = self.forward.CalSurf(model=model, inv_type=inv_type, wtype='lov', solver_str='normal', refc_str=refc_str)
            irefmd = 3

        if 'ray' in wtypes and 'lov' in wtypes:
            irefmd = 4

        # misfit
        # self.Misfit.get_prediction(data_pred)
        # L, L_pre, chiSqr, chiSqr_pre, misfit  = self.Misfit.get_misfit()
        L, L_pre, chiSqr, chiSqr_pre, misfit  = self.Misfit.cal_misfit(data_pred)

        return L, L_pre, chiSqr, chiSqr_pre, misfit, irefmd

    def init_model(self):
        """
        Initialization of parameters 
        mock 
        1) vprofileTTI.update
        2) vprofileTTI.get_vmodel()

        # Initialization
        1) ttimod.readmodtxt() + ttimod.get_paraind() (in read model txt; get c.para
        2) ttimod.update()     # c.para -> vs.para
        3) ttimod.get_vmodel() # vs.para -> layer model; output
        4) ttimod.mod2para()

        """
        # set parameter space
        self.initMod.ttimod.mod2para() # parameter space range (c.para->para space)
        self.initMod.ttimod.define_para_step(earlystage=True) # parameter space, search step
        # update: c.para -> vs.para
        self.initMod.ttimod.update()

        model = self.initMod
        # is good initial model 
        if not model.ttimod.isgood():
            model = self.uniform_walk(model)

        # (vprofileTTI.get_vmodel() get the model arrays and initialize elastic tensor)
        model.set_model()

        return model

    def uniform_walk(self, model):
        """ 
        uniform walk in parameter space

        ttimod.new_paraval: contains ttimod.para2mod() + ttimod.update()
        """
        # assume self.model.ttimod.mod2para() has been done
        if not model.ttimod.new_paraval(ptype=0):
            logger.error(f'New model is problematic Or it does not statisfy the limitations')
            pdb.set_trace()
        model.set_model()
        return model

    def gaussian_walk(self, model):
        """
        input model should keep independence 
        """
        # assume self.model.ttimod.mod2para() has been done
        model_new = copy.deepcopy(model)

        if not model_new.ttimod.new_paraval(ptype=1):
            logger.error(f'New model is problematic Or it does not statisfy the limitations!')
            self.uniform_walk(model_new)
            # pdb.set_trace()

        model_new.set_model()
        return model_new

class Point_TTI(Point):
    def __init__(self, modtype='tti', inv_thickness=False):
        super().__init__(modtype=modtype, inv_thickness=inv_thickness)
    
    def MCinv_tti_old(self, pid=None, Nrun=10000, step4uwalk=2000, misfit_threshold=1.2, NCalL=100, init_run=True, seed=None, 
        verbose=False, priori=False, saveObsdata=False, outdir='MCtest'):
        """
        Bayesian Monte Carlo inversion
        ==========
        self.forward: forward calculation; input model (class), output data (class)
        self.Misfit: calculate misfit; input dataPred (class)

        :: input :: 
        pid         - id for process 
        runN        - total number of run
        step4uwalk  - number for uniform walk
        saveObsdata - save observations
        init_run    - 
        NCalL       - limitation of number of forward calculation
        misfit_threshold     - misfit threshold to shrink step
        ----- 
        History:

        """
        deBug = False 
        random.seed(seed)
        pid = self.pid if pid is None else pid 
        timeStart = time.time()
        kwargs = {}
        # initialization of reference model
        mod0 = self.init_model()

        # initilization of forward
        self.init_misfit_forward()

        # for MC tracking
        npara = mod0.ttimod.para.npara
        mctrack = McTrack(npara=npara, Nrun=Nrun, data=self.dataObs)

        iacc = 0
        shrinkrange = False

        #-------------------------------------------------------------------
        for i in range(Nrun):
            irefmd = 0 # type of reference model
            #-------------------------------------------------------------------
            # termination
            cond1 = self.forward.Ndisp96 > NCalL
            # cond2 = (i > int(Nrun/2.)) and (misfit0 > misfit_threshold)
            if cond1: 
                logger.info(f'max # of calculation of L:{NCalL}! Finished steps: {i:d}'); 
                break
            # if cond2:
            #     logger.info(f'Misfit is too large after half of process! Misfit0:{misfit0:.1f}')
            #     break
            #-------------------------------------------------------------------
            # shrink searching step
            # change the searching step: (Aug 20. 2019) (2020-7-18)
            if (i > Nrun/4) and (misfit0 < misfit_threshold) and not shrinkrange:
                mod0.ttimod.define_para_step(earlystage=False)
                shrinkrange = True
                if verbose:
                    logger.info(f'step {i}, narrower step')
            #-------------------------------------------------------------------
            # initial
            if i % step4uwalk == 0:
                if init_run: 
                    init_run = False 
                else:
                    mod0 = self.uniform_walk(mod0)

                iacc += 1
                L0, LPre0, chiSqr0, chiSqrPre0, misfit0 = self.forward2misfit(mod0, wtype='both', solver_str='normal')

                mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                mctrack.save_data(Misfit=self.Misfit, ind=i)
                if verbose:
                    if init_run: 
                        words = 'starting from reference model'
                    else: 
                        words = 'uniform random walk'
                    logger.info(f'{pid}, step{i:8d}: {words}: likelihood = {L0:.2e},  misfit = {misfit0:.2f}')    

            else: 
                if i % 500.== 0 and verbose:
                    logger.info(f'{pid}, step{i:8d}: current misfit0 = {misfit0:.2f}')
                #-------------------------------------------------------------------    
                # Gaussian walk    
                mod1 = self.gaussian_walk(mod0)
   
                if priori:
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                    mod0 = mod1
                    continue  
                
                L1, LPre1, chiSqr1, chiSqrPre1, misfit1 = self.forward2misfit(mod1, wtype='both', solver_str='perturbation')
                
                #-------------------------------------------------------------------
                # pre-acceptance
                # if not self.accept(chiSqrPre0, chiSqrPre1):
                if not self.accept_likeli(LPre0, LPre1):
                    mctrack.save_model(model=mod1, ind=i, mark_acc=-2, ind_acc=iacc, Misfit=self.Misfit, markKernel=0)
                else: 
                    # 1) check for large dL    
                    kwargs['trigger'] = False
                    if self.forward.eigkR.is_large_dL_perturb(threshold=20.): # 20% perturbation dL; 10% dVs
                        L1, LPre1, chiSqr1, chiSqrPre1, misfit1 = self.forward2misfit(mod1, wtype='ray', solver_str='normal', **kwargs)
                        irefmd = 1

                    if np.isnan(chiSqr1): 
                        logger.error('Error: NaN misfit appears!')
                        pdb.set_trace()
                        continue
                    # 2) final likelihood
                    # if not self.accept(chiSqr0, chiSqr1):
                    if not self.accept_likeli(L0, L1):
                        mctrack.save_model(model=mod1, ind=i, mark_acc=-1, ind_acc=iacc, Misfit=self.Misfit, markKernel=irefmd)
                        mctrack.save_data(Misfit=self.Misfit, ind=i)
                    else: 
                        # check dispersion data
                        # accept model but update reference model
                        reset_wtype = self.forward.check_disp_pertubation() 
                        if len(reset_wtype)>0:
                            L1, LPre1, chiSqr1, chiSqrPre1, misfit1, irefmd = self.reset_forward(mod1, reset_wtype)

                        mod0 = mod1 
                        L0, LPre0, chiSqr0, chiSqrPre0, misfit0 = L1, LPre1, chiSqr1, chiSqrPre1, misfit1
                        iacc += 1
                        mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=irefmd)
                        mctrack.save_data(Misfit=self.Misfit, ind=i)


        # output     
        os.makedirs(outdir, exist_ok=True)
        # MC track   
        mctrack.output_MC(fnm=f'mc_inv.{pid}.npz', outdir=outdir)
        # Observations
        if saveObsdata:
            self.dataObs.save4MC(fnm=f'mc_data.{pid}.npz', outdir=outdir)
        # txt information
        infnm = join(outdir, f'mc_info.{pid}.txt')
        np.savetxt(infnm, [i-1, self.forward.Ndisp96, NCalL, 0], delimiter=" ", fmt='%d')

        if verbose:
            logger.info("time cost = {:.1f} m".format((time.time()-timeStart)/60.))
            num = np.where(mctrack.outmodarr[:,0]==1)[0].size
            logger.info("No. of accepted model: "+str(num))

        del mctrack, mod1, mod0

        return 

    def MCinv_tti(self, pid=None, Nrun=10000, step4uwalk=2000, misfit_threshold=1.2, NCalL=100, init_run=True, seed=None, 
        verbose=False, priori=False, saveObsdata=False, outdir='MCtest'):
        """
        Bayesian Monte Carlo inversion
        New MC chain acceptance.
        ==========
        self.forward: forward calculation; input model (class), output data (class)
        self.Misfit: calculate misfit; input dataPred (class)

        :: input :: 
        pid         - id for process 
        runN        - total number of run
        step4uwalk  - number for uniform walk
        saveObsdata - save observations
        init_run    - 
        NCalL       - limitation of number of forward calculation
        misfit_threshold     - misfit threshold to shrink step
        ----- 
        History:

        """
        deBug = False 
        random.seed(seed)
        pid = self.pid if pid is None else pid 
        timeStart = time.time()
        kwargs = {}
        # initialization of reference model
        mod0 = self.init_model()

        # initilization of forward
        self.init_misfit_forward()

        # for MC tracking
        npara = mod0.ttimod.para.npara
        mctrack = McTrack(npara=npara, Nrun=Nrun, data=self.dataObs)

        iacc = 0
        shrinkrange = False

        #-------------------------------------------------------------------
        for i in range(Nrun):
            irefmd = 0 # type of reference model
            #-------------------------------------------------------------------
            # termination
            cond1 = self.forward.Ndisp96 > NCalL
            # cond2 = (i > int(Nrun/2.)) and (misfit0 > misfit_threshold)
            if cond1: 
                logger.info(f'max # of calculation of L:{NCalL}! Finished steps: {i:d}'); 
                break
            # if cond2:
            #     logger.info(f'Misfit is too large after half of process! Misfit0:{misfit0:.1f}')
            #     break
            #-------------------------------------------------------------------
            # shrink searching step
            # change the searching step: (Aug 20. 2019) (2020-7-18)
            if (i > Nrun/4) and (misfit0 < misfit_threshold) and not shrinkrange:
                mod0.ttimod.define_para_step(earlystage=False)
                shrinkrange = True
                if verbose:
                    logger.info(f'step {i}, narrower step')
            #-------------------------------------------------------------------
            # initial
            if i % step4uwalk == 0:
                if init_run: 
                    init_run = False 
                else:
                    mod0 = self.uniform_walk(mod0)

                iacc += 1
                L0, LPre0, chiSqr0, chiSqrPre0, misfit0 = self.forward2misfit(mod0, wtype='both', solver_str='normal')
                Misfit0 = self.Misfit.copy()

                mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                mctrack.save_data(Misfit=self.Misfit, ind=i)
                if verbose:
                    if init_run: 
                        words = 'starting from reference model'
                    else: 
                        words = 'uniform random walk'
                    logger.info(f'{pid}, step{i:8d}: {words}: likelihood = {L0:.2e},  misfit = {misfit0:.2f}')    

            else: 
                if i % 500.== 0 and verbose:
                    logger.info(f'{pid}, step{i:8d}: current misfit0 = {misfit0:.2f}')
                #-------------------------------------------------------------------    
                # Gaussian walk    
                mod1 = self.gaussian_walk(mod0)
   
                if priori:
                    iacc += 1
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                    mod0 = mod1
                    continue  
                
                L1, LPre1, chiSqr1, chiSqrPre1, misfit1 = self.forward2misfit(mod1, wtype='both', solver_str='perturbation')
                
                #-------------------------------------------------------------------
                # pre-acceptance
                # if not self.accept(chiSqrPre0, chiSqrPre1):
                if not self.accept_likeli(LPre0, LPre1):
                    mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=Misfit0, markKernel=0)
                else: 
                    # 1) check for large dL    
                    kwargs['trigger'] = False
                    if self.forward.eigkR.is_large_dL_perturb(threshold=20.): # 20% perturbation dL; 10% dVs
                        L1, LPre1, chiSqr1, chiSqrPre1, misfit1 = self.forward2misfit(mod1, wtype='ray', solver_str='normal', **kwargs)
                        irefmd = 1

                    if np.isnan(chiSqr1): 
                        logger.error('Error: NaN misfit appears!')
                        pdb.set_trace()
                        continue
                    # 2) final likelihood
                    # if not self.accept(chiSqr0, chiSqr1):
                    if not self.accept_likeli(L0, L1):
                        mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=Misfit0, markKernel=irefmd)
                        mctrack.save_data(Misfit=Misfit0, ind=i)
                    else: 
                        # check dispersion data
                        # accept model but update reference model
                        reset_wtype = self.forward.check_disp_pertubation() 
                        if len(reset_wtype)>0:
                            L1, LPre1, chiSqr1, chiSqrPre1, misfit1, irefmd = self.reset_forward(mod1, reset_wtype)

                        mod0 = mod1 
                        L0, LPre0, chiSqr0, chiSqrPre0, misfit0 = L1, LPre1, chiSqr1, chiSqrPre1, misfit1
                        Misfit0 = self.Misfit.copy()
                        iacc += 1
                        mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=irefmd)
                        mctrack.save_data(Misfit=self.Misfit, ind=i)


        # output     
        os.makedirs(outdir, exist_ok=True)
        # MC track   
        mctrack.output_MC(fnm=f'mc_inv.{pid}.npz', outdir=outdir)
        # Observations
        if saveObsdata:
            self.dataObs.save4MC(fnm=f'mc_data.{pid}.npz', outdir=outdir)
        # txt information
        infnm = join(outdir, f'mc_info.{pid}.txt')
        np.savetxt(infnm, [i-1, self.forward.Ndisp96, NCalL, 0], delimiter=" ", fmt='%d')

        if verbose:
            logger.info("time cost = {:.1f} m".format((time.time()-timeStart)/60.))
            num = np.where(mctrack.outmodarr[:,0]==1)[0].size
            logger.info("No. of accepted model: "+str(num))

        del mctrack, mod1, mod0
        return 

    def MCinv_tti_MP(self, pfx='MC', Nrun=300000, nprocess=20, Nsubrun=10000, step4uwalk=10000, 
         misfit_threshold=1.2, NCalL=200, seed=None, 
         priori=False, deleteSub=True, verbose=True, outdir='MCtest'):
        """ 
        """
        if not exists(outdir):
            os.mkdir(outdir)

        random.seed(seed); seed = random.random()

        if verbose: 
            logger.info('Start MC inversion: '+time.ctime())
            
        logger.info(f"No. of runs: {Nrun}, No. of sub-run: {Nsubrun}, No. of processes: {Nrun//Nsubrun}")
        args = [ [f'MCtemp{i:d}', Nsubrun, step4uwalk, misfit_threshold, NCalL, i==0, seed+i, 
                False, priori, False, outdir] for i in range(Nrun//Nsubrun) ]
        timeStart = time.time() 
        pool = mp.Pool(processes=nprocess)
        pool.starmap(self.MCinv_tti, args)
        pool.close()
        pool.join() 

        # Merge MC output 
        outmodarr, out_ray, out_lov, out_azi, out_amp, out_info = ([] for i in range(6))
        for arg in args:
            pid = arg[0]
            outfnm = join(outdir, f'mc_inv.{pid}.npz')
            tmp = np.load(outfnm, allow_pickle=True)
            outmodarr.append(tmp['outmodarr'])
            out_ray.append(tmp['out_ray'])
            out_lov.append(tmp['out_lov'])
            out_azi.append(tmp['out_azi'])
            out_amp.append(tmp['out_amp'])
            # for info 
            infofnm = join(outdir, f'mc_info.{pid}.txt')
            tmp2 = np.loadtxt(infofnm)
            out_info.append(tmp2)

            if deleteSub:
                os.remove(outfnm)
                os.remove(infofnm)
        # MC info.
        out_info = np.reshape(np.concatenate(out_info), (-1, 4))
        header = "invstep    numCalL     CalLimit    skip#"
        fmt = '%5.0f %5.0f %5.0f %5.0f'
        np.savetxt(join(outdir, f'mc_info.{pfx}.txt'), out_info, delimiter=" ", fmt=fmt, header=header)
        # MC inv tracking
        outmodarr, out_ray, out_lov, out_azi, out_amp = ( np.concatenate(arr, axis=0) for arr in [ outmodarr, out_ray, out_lov, out_azi, out_amp])
        outinvfname = join(outdir, f'mc_inv.{pfx}.npz')
        np.savez_compressed(outinvfname, outmodarr=outmodarr, out_ray=out_ray, out_lov=out_lov, out_azi=out_azi, out_amp=out_amp)
        # MC. observation data
        self.dataObs.save4MC(fnm=f'mc_data.{pfx}.npz', outdir=outdir)

        npara = self.initMod.ttimod.para.npara
        good_threshold = 2.0
        if verbose:
            logger.info('End  MC inversion: '+time.ctime())
            logger.info(f'Elapsed time: {(time.time()-timeStart)/60.:.0f} min')
            ind_acc = np.where(outmodarr[:, 0]==1)[0]
            Nacc = len(ind_acc)
            Ngood = len(np.where(outmodarr[ind_acc, npara+3]<good_threshold)[0])
            logger.info(f"No. of good models: {Ngood} \nNo. of total model: {Nacc}")
            logger.info('---------- Finish M.C. ---------------')
        return 

class Point_HTI_Joint(Point):

    def __init__(self, modtype='hti', inv_thickness=False, hti_azi_only=False):
        super().__init__(modtype=modtype,inv_thickness=inv_thickness, hti_azi_only=hti_azi_only)

    def MCinv(self, pid=None, Nrun=10000, step4uwalk=2000, misfit_threshold=1.2, NCalL=100, init_run=True, seed=None, 
        verbose=False, priori=False, saveObsdata=False, outdir='MCtest'):
        """
        Bayesian Monte Carlo inversion
        New MC chain acceptance.
        ==========
        self.forward: forward calculation; input model (class), output data (class)
        self.Misfit: calculate misfit; input dataPred (class)

        :: input :: 
        pid         - id for process 
        runN        - total number of run
        step4uwalk  - number for uniform walk
        saveObsdata - save observations
        init_run    - 
        NCalL       - limitation of number of forward calculation
        misfit_threshold     - misfit threshold to shrink step
        ----- 
        History:

        """
        deBug = False 
        random.seed(seed)
        pid = self.pid if pid is None else pid 
        timeStart = time.time()
        kwargs = {}
        # initialization of reference model
        mod0 = self.init_model()

        # initilization of forward
        self.init_misfit_forward()

        # for MC tracking
        npara = mod0.ttimod.para.npara
        mctrack = McTrack(npara=npara, Nrun=Nrun, data=self.dataObs)

        iacc = 0
        shrinkrange = False

        #-------------------------------------------------------------------
        for i in range(Nrun):
            irefmd = 0 # type of reference model
            #-------------------------------------------------------------------
            # termination
            cond1 = self.forward.Ndisp96 > NCalL
            if cond1: 
                logger.info(f'max # of calculation of L:{NCalL}! Finished steps: {i:d}'); 
                break
            #-------------------------------------------------------------------
            # shrink searching step
            # change the searching step: (Aug 20. 2019) (2020-7-18)
            if (i > Nrun/4) and (misfit0 < misfit_threshold) and not shrinkrange:
                mod0.ttimod.define_para_step(earlystage=False)
                shrinkrange = True
                if verbose:
                    logger.info(f'step {i}, narrower step')
            #-------------------------------------------------------------------
            # initial
            if i % step4uwalk == 0:
                if init_run: 
                    init_run = False 
                else:
                    mod0 = self.uniform_walk(mod0)

                iacc += 1
                L0, LPre0, chiSqr0, chiSqrPre0, misfit0 = self.forward2misfit(model=mod0, inv_type='hti', wtype='ray', solver_str='normal')
                Misfit0 = self.Misfit.copy()

                mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                mctrack.save_data(Misfit=self.Misfit, ind=i)
                if verbose:
                    if init_run: 
                        words = 'starting from reference model'
                    else: 
                        words = 'uniform random walk'
                    logger.info(f'{pid}, step{i:8d}: {words}: likelihood = {L0:.2e},  misfit = {misfit0:.2f}')    

            else: 
                if i % 500.== 0 and verbose:
                    logger.info(f'{pid}, step{i:8d}: current misfit0 = {misfit0:.2f}')
                #-------------------------------------------------------------------    
                # Gaussian walk    
                mod1 = self.gaussian_walk(mod0)
   
                if priori:
                    iacc += 1
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                    mod0 = mod1
                    continue  
                
                L1, LPre1, chiSqr1, chiSqrPre1, misfit1 = self.forward2misfit(model=mod1, inv_type='hti', wtype='ray', solver_str='perturbation')
                #-------------------------------------------------------------------
                # pre-acceptance
                # if not self.accept(chiSqrPre0, chiSqrPre1):
                if not self.accept_likeli(LPre0, LPre1):
                    mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=Misfit0, markKernel=0)
                else: 
                    # 1) check for large dL    
                    kwargs['trigger'] = False
                    if self.forward.eigkR.is_large_dL_perturb(threshold=20.): # 20% perturbation dL; 10% dVs
                        L1, LPre1, chiSqr1, chiSqrPre1, misfit1 = self.forward2misfit(model=mod1, inv_type='hti', wtype='ray', solver_str='normal', **kwargs)
                        irefmd = 1

                    if np.isnan(chiSqr1): 
                        logger.error('Error: NaN misfit appears!')
                        pdb.set_trace()
                        continue
                    # 2) final likelihood
                    # if not self.accept(chiSqr0, chiSqr1):
                    if not self.accept_likeli(L0, L1):
                        mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=Misfit0, markKernel=irefmd)
                        mctrack.save_data(Misfit=Misfit0, ind=i)
                    else: 
                        # check dispersion data
                        # accept model but update reference model
                        reset_wtype = self.forward.check_disp_pertubation() 
                        if len(reset_wtype)>0:
                            L1, LPre1, chiSqr1, chiSqrPre1, misfit1, irefmd = self.reset_forward(mod1, reset_wtype)

                        mod0 = mod1 
                        L0, LPre0, chiSqr0, chiSqrPre0, misfit0 = L1, LPre1, chiSqr1, chiSqrPre1, misfit1
                        Misfit0 = self.Misfit.copy()
                        iacc += 1
                        mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=irefmd)
                        mctrack.save_data(Misfit=self.Misfit, ind=i)


        # output     
        os.makedirs(outdir, exist_ok=True)
        # MC track   
        mctrack.output_MC(fnm=f'mc_inv.{pid}.npz', outdir=outdir)
        # Observations
        if saveObsdata:
            self.dataObs.save4MC(fnm=f'mc_data.{pid}.npz', outdir=outdir)
        # txt information
        infnm = join(outdir, f'mc_info.{pid}.txt')
        np.savetxt(infnm, [i-1, self.forward.Ndisp96, NCalL, 0], delimiter=" ", fmt='%d')

        if verbose:
            logger.info("time cost = {:.1f} m".format((time.time()-timeStart)/60.))
            num = np.where(mctrack.outmodarr[:,0]==1)[0].size
            logger.info("No. of accepted model: "+str(num))

        del mctrack, mod1, mod0
        return 

    def MCinv_MP(self, pfx='MC', Nrun=300000, nprocess=20, Nsubrun=10000, step4uwalk=10000, 
         misfit_threshold=1.2, NCalL=200, seed=None, 
         priori=False, deleteSub=True, verbose=True, outdir='MCtest'):
        """ 
        """
        if not exists(outdir):
            os.mkdir(outdir)

        random.seed(seed); seed = random.random()

        if verbose: 
            logger.info('Start MC inversion: '+time.ctime())
            
        logger.info(f"No. of runs: {Nrun}, No. of sub-run: {Nsubrun}, No. of processes: {Nrun//Nsubrun}")
        args = [ [f'MCtemp{i:d}', Nsubrun, step4uwalk, misfit_threshold, NCalL, i==0, seed+i, 
                False, priori, False, outdir] for i in range(Nrun//Nsubrun) ]
        timeStart = time.time() 
        pool = mp.Pool(processes=nprocess)
        pool.starmap(self.MCinv, args)
        pool.close()
        pool.join() 

        # Merge MC output 
        outmodarr, out_ray, out_azi, out_amp, out_info = ([] for i in range(5))
        for arg in args:
            pid = arg[0]
            outfnm = join(outdir, f'mc_inv.{pid}.npz')
            tmp = np.load(outfnm, allow_pickle=True)
            outmodarr.append(tmp['outmodarr'])
            out_ray.append(tmp['out_ray'])
            out_azi.append(tmp['out_azi'])
            out_amp.append(tmp['out_amp'])
            # for info 
            infofnm = join(outdir, f'mc_info.{pid}.txt')
            tmp2 = np.loadtxt(infofnm)
            out_info.append(tmp2)

            if deleteSub:
                os.remove(outfnm)
                os.remove(infofnm)
        # MC info.
        out_info = np.reshape(np.concatenate(out_info), (-1, 4))
        header = "invstep    numCalL     CalLimit    skip#"
        fmt = '%5.0f %5.0f %5.0f %5.0f'
        np.savetxt(join(outdir, f'mc_info.{pfx}.txt'), out_info, delimiter=" ", fmt=fmt, header=header)
        # MC inv tracking
        outmodarr, out_ray, out_azi, out_amp = ( np.concatenate(arr, axis=0) for arr in [ outmodarr, out_ray, out_azi, out_amp])
        outinvfname = join(outdir, f'mc_inv.{pfx}.npz')
        np.savez_compressed(outinvfname, outmodarr, out_ray, out_azi, out_amp)
        # MC. observation data
        self.dataObs.save4MC(fnm=f'mc_data.{pfx}.npz', outdir=outdir)

        npara = self.initMod.ttimod.para.npara
        good_threshold = 2.0
        if verbose:
            logger.info('End  MC inversion: '+time.ctime())
            logger.info(f'Elapsed time: {(time.time()-timeStart)/60.:.0f} min')
            ind_acc = np.where(outmodarr[:, 0]==1)[0]
            Nacc = len(ind_acc)
            Ngood = len(np.where(outmodarr[ind_acc, npara+3]<good_threshold)[0])
            logger.info(f"No. of good models: {Ngood} \nNo. of total model: {Nacc}")
            logger.info('---------- Finish M.C. ---------------')
        return 

class Point_HTI(Point):

    def __init__(self, modtype='hti', inv_thickness=False, hti_azi_only=False):
        super().__init__(modtype=modtype, inv_thickness=inv_thickness, hti_azi_only=hti_azi_only)

    def MCinv(self, pid=None, Nrun=10000, step4uwalk=10000, misfit_threshold=1.2, NCalL=100, init_run=False, seed=None, 
        verbose=False, priori=False, saveObsdata=False, outdir='MCtest'):
        """
        Bayesian Monte Carlo inversion
        New MC chain acceptance.
        ==========
        self.forward: forward calculation; input model (class), output data (class)
        self.Misfit: calculate misfit; input dataPred (class)

        :: input :: 
        pid         - id for process 
        runN        - total number of run
        step4uwalk  - number for uniform walk
        saveObsdata - save observations
        init_run    - 
        NCalL       - limitation of number of forward calculation
        misfit_threshold     - misfit threshold to shrink step
        ----- 
        History:

        """
        deBug = False 
        random.seed(seed)
        pid = self.pid if pid is None else pid 
        timeStart = time.time()
        kwargs = {}
        # initialization of reference model
        mod0 = self.init_model()

        # initilization of forward
        self.init_misfit_forward()

        # for MC tracking
        npara = mod0.ttimod.para.npara
        mctrack = McTrack(npara=npara, Nrun=Nrun, data=self.dataObs)

        iacc = 0
        shrinkrange = False

        #-------------------------------------------------------------------
        for i in range(Nrun):
            irefmd = 0 # type of reference model
            #-------------------------------------------------------------------
            #-------------------------------------------------------------------
            # shrink searching step
            # change the searching step: (Aug 20. 2019) (2020-7-18)
            if (i > Nrun/4) and (misfit0 < misfit_threshold) and not shrinkrange:
                mod0.ttimod.define_para_step(earlystage=False)
                shrinkrange = True
                if verbose:
                    logger.info(f'step {i}, narrower step')
            #-------------------------------------------------------------------
            # initial
            if i % step4uwalk == 0:
                if init_run: 
                    init_run = False 
                else:
                    mod0 = self.uniform_walk(mod0)

                iacc += 1
                L0, _, chiSqr0, _, misfit0 = self.forward2misfit(model=mod0, inv_type='hti', wtype='ray', solver_str='normal')
                Misfit0 = self.Misfit.copy()

                mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                mctrack.save_data(Misfit=self.Misfit, ind=i)
                if verbose:
                    if init_run: 
                        words = 'starting from reference model'
                    else: 
                        words = 'uniform random walk'
                    logger.info(f'{pid}, step{i:8d}: {words}: likelihood = {L0:.2e},  misfit = {misfit0:.2f}')    

            else: 
                if i % 500.== 0 and verbose:
                    logger.info(f'{pid}, step{i:8d}: current misfit0 = {misfit0:.2f}')
                #-------------------------------------------------------------------    
                # Gaussian walk    
                mod1 = self.gaussian_walk(mod0)
   
                if priori:
                    iacc += 1
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                    mod0 = mod1
                    continue  
                
                L1, _, chiSqr1, _, misfit1 = self.forward2misfit(model=mod1, inv_type='hti', wtype='ray', solver_str='perturbation')
                #-------------------------------------------------------------------
                if not self.accept_likeli(L0, L1):
                    mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=Misfit0, markKernel=irefmd)
                    mctrack.save_data(Misfit=Misfit0, ind=i)
                else: 
                    mod0 = mod1 
                    L0, chiSqr0, misfit0 = L1, chiSqr1, misfit1
                    Misfit0 = self.Misfit.copy()
                    iacc += 1
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=irefmd)
                    mctrack.save_data(Misfit=self.Misfit, ind=i)

        # output     
        os.makedirs(outdir, exist_ok=True)
        # MC track   
        mctrack.output_MC(fnm=f'mc_inv.{pid}.npz', outdir=outdir)
        # Observations
        if saveObsdata:
            self.dataObs.save4MC(fnm=f'mc_data.{pid}.npz', outdir=outdir)

        if verbose:
            logger.info("time cost = {:.1f} m".format((time.time()-timeStart)/60.))
            num = np.where(mctrack.outmodarr[:,0]==1)[0].size
            logger.info("No. of accepted model: "+str(num))

        del mctrack, mod1, mod0
        return 

    def MCinv_MP(self, pfx='MC', Nrun=300000, nprocess=20, Nsubrun=10000, step4uwalk=10000, 
         misfit_threshold=1.2, NCalL=200, seed=None, 
         priori=False, deleteSub=True, verbose=True, outdir='MCtest'):
        """ 
        """
        if not exists(outdir):
            os.mkdir(outdir)

        random.seed(seed); seed = random.random()

        if verbose: 
            logger.info('Start MC inversion: '+time.ctime())
            
        logger.info(f"No. of runs: {Nrun}, No. of sub-run: {Nsubrun}, No. of processes: {Nrun//Nsubrun}")
        args = [ [f'MCtemp{i:d}', Nsubrun, step4uwalk, misfit_threshold, NCalL, i==0, seed+i, 
                False, priori, False, outdir] for i in range(Nrun//Nsubrun) ]
        timeStart = time.time() 
        pool = mp.Pool(processes=nprocess)
        pool.starmap(self.MCinv, args)
        pool.close()
        pool.join() 

        # Merge MC output 
        outmodarr, out_ray, out_azi, out_amp, out_info = ([] for i in range(5))
        for arg in args:
            pid = arg[0]
            outfnm = join(outdir, f'mc_inv.{pid}.npz')
            tmp = np.load(outfnm, allow_pickle=True)
            outmodarr.append(tmp['outmodarr'])
            out_ray.append(tmp['out_ray'])
            out_azi.append(tmp['out_azi'])
            out_amp.append(tmp['out_amp'])

            if deleteSub:
                os.remove(outfnm)

        # MC inv tracking
        outmodarr, out_ray, out_azi, out_amp = ( np.concatenate(arr, axis=0) for arr in [ outmodarr, out_ray, out_azi, out_amp])
        outinvfname = join(outdir, f'mc_inv.{pfx}.npz')
        np.savez_compressed(outinvfname, outmodarr=outmodarr, out_ray=out_ray, out_azi=out_azi, out_amp=out_amp)

        # MC. observation data
        self.dataObs.save4MC(fnm=f'mc_data.{pfx}.npz', outdir=outdir)

        npara = self.initMod.ttimod.para.npara
        good_threshold = 1.5
        if verbose:
            logger.info('End  MC inversion: '+time.ctime())
            logger.info(f'Elapsed time: {(time.time()-timeStart)/60.:.0f} min')
            ind_acc = np.where(outmodarr[:, 0]==1)[0]
            Nacc = len(ind_acc)
            Ngood = len(np.where(outmodarr[ind_acc, npara+3]<good_threshold)[0])
            logger.info(f"No. of good models: {Ngood} \nNo. of total model: {Nacc}")
            logger.info('---------- Finish M.C. ---------------')
        return 

class Point_ISO(Point):

    def __init__(self, modtype='iso', inv_thickness=False):
        super().__init__(modtype=modtype, inv_thickness=inv_thickness)

    def MCinv(self, pid=None, Nrun=10000, step4uwalk=2000, misfit_threshold=1.4, init_run=True, seed=None, 
        verbose=False, priori=False, saveObsdata=False, outdir='MCtest'):
        """
        Bayesian Monte Carlo inversion
        New MC chain acceptance.
        ==========
        self.forward: forward calculation; input model (class), output data (class)
        self.Misfit: calculate misfit; input dataPred (class)

        :: input :: 
        pid         - id for process 
        runN        - total number of run
        step4uwalk  - number for uniform walk
        saveObsdata - save observations
        ----- 
        History:

        """
        deBug = False 
        random.seed(seed)
        pid = self.pid if pid is None else pid 
        timeStart = time.time()
        kwargs = {}
        # initialization of reference model
        mod0 = self.init_model()

        # initilization of forward
        self.init_misfit_forward()

        # for MC tracking
        npara = mod0.ttimod.para.npara
        mctrack = McTrack(npara=npara, Nrun=Nrun, data=self.dataObs)

        iacc = 0
        shrinkrange = False

        #-------------------------------------------------------------------
        for i in range(Nrun):
            irefmd = 0 # type of reference model

            #-------------------------------------------------------------------
            # initial
            if i % step4uwalk == 0:
                if init_run: 
                    init_run = False 
                else:
                    mod0 = self.uniform_walk(mod0)

                iacc += 1
                L0, _, chiSqr0, _, misfit0 = self.forward2misfit(model=mod0, inv_type='iso', wtype='ray')
                Misfit0 = self.Misfit.copy()

                mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                mctrack.save_data(Misfit=self.Misfit, ind=i)
                if verbose:
                    if init_run: 
                        words = 'starting from reference model'
                    else: 
                        words = 'uniform random walk'
                    logger.info(f'{pid}, step{i:8d}: {words}: likelihood = {L0:.2e},  misfit = {misfit0:.2f}')    

            else: 
                if i % 500.== 0 and verbose:
                    logger.info(f'{pid}, step{i:8d}: current misfit0 = {misfit0:.2f}')
                #-------------------------------------------------------------------    
                # Gaussian walk    
                mod1 = self.gaussian_walk(mod0)
   
                if priori:
                    iacc += 1
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=1)
                    mod0 = mod1
                    continue  
                
                L1, _, chiSqr1, _, misfit1 = self.forward2misfit(model=mod1, inv_type='iso', wtype='ray')
                #-------------------------------------------------------------------
                if not self.accept_likeli(L0, L1):
                    mctrack.save_model(model=mod0, ind=i, mark_acc=1, ind_acc=iacc, Misfit=Misfit0, markKernel=irefmd)
                    mctrack.save_data(Misfit=Misfit0, ind=i)
                else: 
                    mod0 = mod1 
                    L0, chiSqr0, misfit0 = L1, chiSqr1, misfit1
                    Misfit0 = self.Misfit.copy()
                    iacc += 1
                    mctrack.save_model(model=mod1, ind=i, mark_acc=1, ind_acc=iacc, Misfit=self.Misfit, markKernel=irefmd)
                    mctrack.save_data(Misfit=self.Misfit, ind=i)

        # output     
        os.makedirs(outdir, exist_ok=True)
        # MC track   
        mctrack.output_MC(fnm=f'mc_inv.{pid}.npz', outdir=outdir)
        # Observations
        if saveObsdata:
            self.dataObs.save4MC(fnm=f'mc_data.{pid}.npz', outdir=outdir)

        if verbose:
            logger.info("time cost = {:.1f} m".format((time.time()-timeStart)/60.))
            num = np.where(mctrack.outmodarr[:,0]==1)[0].size
            logger.info("No. of accepted model: "+str(num))

        del mctrack, mod1, mod0
        return 

    def MCinv_MP(self, pfx='MC', Nrun=200000, nprocess=20, Nsubrun=10000, step4uwalk=10000, 
         seed=None, priori=False, deleteSub=True, verbose=True, outdir='MCtest'):
        """ 
        """
        if not exists(outdir):
            os.mkdir(outdir)

        random.seed(seed); seed = random.random()

        if verbose: 
            logger.info('Start MC inversion: '+time.ctime())
            
        logger.info(f"No. of runs: {Nrun}, No. of sub-run: {Nsubrun}, No. of processes: {Nrun//Nsubrun}")
        args = [ [f'MCtemp{i:d}', Nsubrun, step4uwalk, i==0, seed+i, False, priori, False, outdir] for i in range(Nrun//Nsubrun) ]
        timeStart = time.time() 
        pool = mp.Pool(processes=nprocess)
        pool.starmap(self.MCinv, args)
        pool.close()
        pool.join() 

        # Merge MC output 
        outmodarr, out_ray = ([] for i in range(2))
        for arg in args:
            pid = arg[0]
            outfnm = join(outdir, f'mc_inv.{pid}.npz')
            tmp = np.load(outfnm, allow_pickle=True)
            outmodarr.append(tmp['outmodarr'])
            out_ray.append(tmp['out_ray'])

            if deleteSub:
                os.remove(outfnm)

        # MC inv tracking
        outmodarr, out_ray = ( np.concatenate(arr, axis=0) for arr in [outmodarr, out_ray])
        outinvfname = join(outdir, f'mc_inv.{pfx}.npz')
        np.savez_compressed(outinvfname, outmodarr=outmodarr, out_ray=out_ray)
        # MC. observation data
        self.dataObs.save4MC(fnm=f'mc_data.{pfx}.npz', outdir=outdir)

        npara = self.initMod.ttimod.para.npara
        good_threshold = 1.5
        if verbose:
            logger.info('End  MC inversion: '+time.ctime())
            logger.info(f'Elapsed time: {(time.time()-timeStart)/60.:.0f} min')
            ind_acc = np.where(outmodarr[:, 0]==1)[0]
            Nacc = len(ind_acc)
            Ngood = len(np.where(outmodarr[ind_acc, npara+3]<good_threshold)[0])
            logger.info(f"No. of good models: {Ngood} \n No. of total model: {Nacc}")
            logger.info('---------- Finish M.C. ---------------')
        return 

def Set_point_inv(modtype, **kwargs):

    hti_azi_only = kwargs.get('only_azi_data', False)

    if modtype == 'iso':
        point = Point_ISO(modtype=modtype)
    elif modtype == 'hti':
        point = Point_HTI(modtype=modtype, hti_azi_only=hti_azi_only)

    elif modtype == 'tti':
        point = Point_TTI(modtype=modtype)
    return point


def get_datafile(datapath, keystr):
    
    fdispR = join(datapath, f'dispISO_R_{keystr}.txt')
    fdispL = join(datapath, f'dispISO_L_{keystr}.txt')
    fazi_phi = join(datapath, f'dispPHI_{keystr}.txt')
    fazi_amp = join(datapath, f'dispAMP_{keystr}.txt')
    
    return  fdispR, fdispL, fazi_phi, fazi_amp

def tti_inv_test():
    pointlabel = '-157.0_61.5'
    datapath = '/home/liu/project/AK_tti/data/data_2022_07_18'
    modpath = '/home/liu/project/AK_tti/data/mean_mod_4MC_2022_06_23'

    # load data
    pinv = Point_TTI()
    fdispR, fdispL, fazi_amp, fazi_phi = get_datafile(datapath, pointlabel) 
    pinv.readObsData(dtype='dispR', infnm=fdispR)
    pinv.readObsData(dtype='dispL', infnm=fdispL)
    pinv.readObsData(dtype='dispPhi', infnm=fazi_phi)
    pinv.readObsData(dtype='dispAmp', infnm=fazi_amp)
    # load mod 
    modfnm = glob.glob(join(modpath, f'*{pointlabel}*'))[0]  
    pinv.readmod(modfnm=modfnm, elltype=True)

    # MC inv 
    # pinv.MCinv_tti(pid='MCtemp1', Nrun=10000, step4uwalk=2000, misfit_threshold=1.2, NCalL=200, init_run=False, seed=20, 
    #     verbose=True, priori=False, savedata=True, outdir='./MCtest')
    Nrun = 300000
    nprocess = 15
    Nsubrun = int(Nrun/nprocess)
    pinv.MCinv_tti_MP(Nrun=300000, nprocess=30, Nsubrun=Nsubrun, step4uwalk=Nsubrun, NCalL=100, 
    verbose=True, outdir='./MCtest')
    
    # postp = PostPoint('MCtest/test.npz','MCtest_priori/test.npz')
    # postp.plotDisp()
    # postp.plotDistrib()
    # fig = postp.plotVsProfileGrid()
    # mod2.plotProfileGrid(fig=fig,label='True',lineStyle='--')
    # plt.legend()



def set_logging(keyword='MC_test'):
    log_format = '[%(asctime)-12s] %(levelname)-8s - %(message)s'
    log_date_format = '%m-%d %H:%M'
    level = logging.INFO

    fnm_log = f"MC_{keyword}.log"

    logging.basicConfig(filename=fnm_log, filemode='a', format=log_format, datefmt=log_date_format, level=level)
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

if __name__ == '__main__': 
    set_logging()

