import sys,os,time
from math import sqrt, floor, ceil
import cPickle as pickle
import numpy as np
import pandas as pd
import argparse

opt = None

################ useful stuff ###############

def combine_dicts(*dicts):
    tuples = []
    for i in dicts:
        if i is not None:
            tuples += i.items()
    return dict(tuples) 

def getOrElse(map, key, default=None):
    if key not in map or map[key] is None:
        return default
    else:
        return map[key]


def init_params(argv):
    global opt
   
    # parse command line
    parser = argparse.ArgumentParser()

    parser.add_argument('-ID', type=str, nargs='+', default=["test"], help="id strings for this run")

    parser.add_argument('-words', type=str, default="words_2018-03-19.pkl")

    parser.add_argument('-humans', type=str, default="humans_7-14-2018.pkl") 
    parser.add_argument('-hid', type=str, nargs='+', default=[None])
    parser.add_argument('-start_net', dest='start_net', type=str, nargs='+', default=None)

    parser.add_argument('-params', dest='params', type=str, default="treatment.params.pkl")
    parser.add_argument('-pi', type=int, nargs='+')

    
    parser.add_argument('-treatment_run', action='store_true', default=False, help="Is this a treatment run?")
    parser.add_argument('-treatments', dest='treatments', type=str, default=None)
    parser.add_argument('-ti', type=int, nargs='+')

    parser.add_argument('-zip', action='store_true', default=False, help="zip pi and hid lists (don't run all combinations)")

    parser.add_argument('-lesion', action='store_true', default=False, help="Is this a lesion run?")
    parser.add_argument('-treatment_language', type=str, default=None)

    parser.add_argument('-flip_language', action='store_true', default=False, help="Flip the treatment language")
    
    parser.add_argument('-run', type=int, default=None)
    parser.add_argument('-nruns', type=int, default=1)  ### obsolete -Uli 7/17/18

    parser.add_argument('-naming_only', action='store_true', default=False, help="train naming assoc only")
    parser.add_argument('-norm', type=str, default='classic')
    parser.add_argument('-hood_std_devs', type=float, default=2.0, help="neighborhood size in stdevs of gaussian sigma function")

    parser.add_argument('-nbatch', type=int, default=8)    
    parser.add_argument('-maxR', type=float, default=10)
    parser.add_argument('-minR', type=float, default=3)
    parser.add_argument('-act_fun', type=str, default='gauss')

    parser.add_argument('-train_across_method', type=str, default='p')

    parser.add_argument('-sem_lesion_strength', type=float, default = 0.0)
    parser.add_argument('-eng_lesion_strength', type=float, default = 0.0)
    parser.add_argument('-spa_lesion_strength', type=float, default = 0.0)
    
    parser.add_argument('-pre_seed', type=int, default = -1)
    
    parser.add_argument('-sem_lesion_seed', type=int, default = -1)
    parser.add_argument('-eng_lesion_seed', type=int, default = -1)
    parser.add_argument('-spa_lesion_seed', type=int, default = -1)
    parser.add_argument('-post_seed', type=int, default = -1)

    parser.add_argument('-noleak', action='store_true', default=False, help="Don't leak, duh") 

    parser.add_argument('-force_mono', action='store_true', default=True, help="force alpha and sigma do decrease after age 10")
    
    parser.add_argument('-mask_assoc', action='store_true', default=False, help="Don't apply mask to assoc connections when lesioning") 
    
    parser.add_argument('-treatment_seed', type=int, default = -1)

    parser.add_argument('-test_freq', type=float, default=1.0, help="test interval")
    parser.add_argument('-no_trans', action='store_true', default=False, help="Don't test translation") 
    parser.add_argument('-initial_test', action='store_true', default=False, help="Test before we do anything")

    parser.add_argument('-save_pre', action='store_true', default=False, help="save final net")
    parser.add_argument('-save_post', action='store_true', default=False, help="save final net")
    parser.add_argument('-save_pre_treat', action='store_true', default=False, help="save final net")
    parser.add_argument('-save_post_treat', action='store_true', default=False, help="save final net")
    parser.add_argument('-save_prefix', type=str, default=None)


    parser.add_argument('-save_outputs', action='store_true', default=False, help="save raw test outputs")

    parser.add_argument('-nogpu', action='store_true', default=False, help="Don't run model on GPU")
    parser.add_argument('-nice', action='store_true', default=False, help="Don't hog all GPU resources")

    opt = parser.parse_args(argv)

    return opt



    

################## PARAMETERS #########################

class P(object):

    file_name = None
    frame = None
    pi = None
    params = None

    hid = None
    start_net = None

    pre_seed = None
    sem_lesion_seed = None
    eng_lesion_seed = None
    spa_lesion_seed = None
    post_seed = None   
    
    map_size = None
    exposure = None
    min_exp = None
    act_fun = None
    nbatch = None
    alpha_years = None
    alphas = None
    assoc_alpha_factor = None
    sigma_years = None
    sigmas = None
    maxR = None
    minR=None

    norm = None
    noise = None
    noise_std = None

    hood_std_devs = None
    train_across_method = None

    
    lesion_run = None
    treatment_run = None

    semantic_lesion_type = None
    semantic_lesion_strength = None
    assoc_lesion_type = None
    assoc_lesion_strength = None
    phonetic_lesion_type = None
    eng_lesion_strength = None
    spa_lesion_strength = None

    mask_assoc = None
    
    min_word_freq = None

          
    
    @classmethod
    def load(cls, fname):
        cls.frame = pd.read_pickle(fname)
        cls.file_name = fname
        
    @classmethod
    def getp(cls, name, default=None):
        if name in cls.params:
            return cls.params[name]
        elif name.upper() in cls.params:
            return cls.params[name.upper()]       
        elif name.lower() in cls.params:
            return cls.params[name.lower()]
        elif default is not None:       
            return default
        else:
            return None

        
    @classmethod
    def sigmaf(cls):
        return lambda age: np.interp(age, cls.sigma_years, cls.sigmas)  

    @classmethod
    def map_alphaf(cls):    
        return lambda age: np.interp(age, cls.alpha_years, cls.alphas)
    
    @classmethod
    def assoc_alphaf(cls):
        return lambda age: cls.assoc_alpha_factor * np.interp(age, cls.alpha_years, cls.alphas)   
          
                
    @classmethod
    def set_pi(cls, pi):

        cls.pi = pi
        cls.params = cls.frame.loc[pi].copy()

        cls.hid = cls.getp("hid", opt.hid)
        cls.start_net = cls.getp("start_net", opt.start_net)

        cls.lesion_run = cls.getp("lesion_run", opt.lesion)
        cls.treatment_run = cls.getp("treatment_run", opt.treatment_run)
        
        cls.pre_seed = cls.getp("pre_seed", opt.pre_seed)
        cls.sem_lesion_seed = cls.getp("sem_lesion_seed", opt.sem_lesion_seed)
        cls.eng_lesion_seed = cls.getp("eng_lesion_seed", opt.eng_lesion_seed)
        cls.spa_lesion_seed = cls.getp("spa_lesion_seed", opt.spa_lesion_seed)
        cls.post_seed = cls.getp("post_seed", opt.post_seed)
                       
        #cls.treatment_seed = cls.getp("treatment_seed", opt.treatment_seed)
        
        cls.map_size = int(cls.getp("MAP_SIZE"))
        cls.exposure = cls.getp("EXPOSURE", 1.0)
        cls.min_exp = cls.getp("MIN_EXP", 0.0)

        cls.maxR = cls.getp("maxR", opt.maxR)
        cls.minR = cls.getp("minR", opt.minR)
        cls.act_fun = cls.getp("ACT_FUN", opt.act_fun)
        cls.nbatch = cls.getp("NBATCH", opt.nbatch)

        cls.alpha_years = cls.params.index[cls.params.index.str.startswith("ALPHA_")].map(lambda col: int(col[6:])).tolist()
        cls.alpha_years.sort()
        ALPHAS = ["ALPHA_%d"%year for year in cls.alpha_years]
        if True: #opt.force_mono:
            cls.params[ALPHAS[2:]] = cls.params[ALPHAS[2:]].cummin()
        
        cls.alphas = cls.params[ALPHAS].tolist()

        cls.assoc_alpha_factor = cls.getp("assoc_alpha_factor", default=1.0)

        cls.train_across_method = cls.getp("train_across_method", default=opt.train_across_method)  #"p","q","or", "and"

        
        cls.sigma_years = cls.params.index[cls.params.index.str.startswith("SIGMA_")].map(lambda col: int(col[6:])).tolist()
        cls.sigma_years.sort()
        SIGMAS = ["SIGMA_%d"%year for year in cls.sigma_years]
        if True: #opt.force_mono:
            cls.params[SIGMAS[2:]] = cls.params[SIGMAS[2:]].cummin()
        cls.sigmas = cls.params[SIGMAS].tolist()

        cls.noise_std = cls.getp("NOISE_STD", 0.0)
        cls.noise = cls.getp("NOISE", "uniform")
        cls.norm = cls.getp("NORM", default=opt.norm)        

        cls.semantic_lesion_type = cls.getp("SEM_LESION_TYPE", default="none")
        cls.semantic_lesion_strength = cls.getp("SEM_LESION_STRENGTH", default=opt.sem_lesion_strength)
        cls.phonetic_lesion_type = cls.getp("PHONETIC_LESION_TYPE", default="none")
        cls.eng_lesion_strength = cls.getp("ENG_LESION_STRENGTH", default=opt.eng_lesion_strength)
        cls.spa_lesion_strength = cls.getp("SPA_LESION_STRENGTH", default=opt.spa_lesion_strength)

        cls.mask_assoc = cls.getp("mask_assoc", default=opt.mask_assoc)
        
        cls.papt_thresh = cls.getp("PAPT_THRESH", default=0.2)
        cls.min_word_freq = cls.getp("MIN_WORD_FREQ",default=0.0)

        cls.hood_std_devs =  cls.getp("HOOD_STD_DEVS", opt.hood_std_devs)

        return pi




################## TREATMENT PARAMS #########################


class T(object):

    file_name = None
    frame = None
    ti = None
    treatment = None

    treatment_seed = -1

    select_words = None
    ntreatment_words = None

    treatment_sem_factor = None 
    treatment_alpha_factor = None

    noleak = None
    leak_filter = None
    treatment_leak_factor = None
    treatment_leak_factor2 = None
    
    treatment_hood_size = None

    assoc_train_method = None

    sessions_per_week = None
    treatment_normal_exp_weeks = None
    
    leak_named_only = None
    leak_alt_naming = None
    leak_train_map_method = None
    leak_train_map_factor = None

    leak_use_fake_act = None

    
    @classmethod
    def load(cls, fname):
        cls.frame = pickle.load(open(fname))
        cls.file_name = fname
        
    @classmethod
    def getp(cls, name, default=None):
        if name in cls.treatment:
            return cls.treatment[name]        
        elif name.upper() in cls.treatment:
            return cls.treatment[name.upper()]       
        elif name.lower() in cls.treatment:
            return cls.treatment[name.lower()]
        else:       
            return default

        
    @classmethod
    def set_ti(cls, ti):

        cls.ti = ti
        cls.treatment = cls.frame.ix[ti]
      
        cls.seed = cls.getp("treatment_seed", opt.treatment_seed)
        
        cls.ntreatment_words = cls.getp("ntreatment_words", 30)
        
        cls.treatment_sem_factor = cls.getp("treatment_sem_factor", 0.5)
        cls.treatment_alpha_factor = cls.getp("treatment_alpha_factor", 0.5)


        cls.noleak = cls.getp("noleak", opt.noleak)
        
        cls.leak_filter = cls.getp("leak_filter", "trans")  ### trans, named, none
        
        cls.leak_named_only = cls.getp("leak_named_only", True)  ### obsolete
        cls.leak_alt_naming = cls.getp("leak_alt_naming", True)  ### obsolete
        
        cls.treatment_leak_factor = cls.getp("treatment_leak_factor", 0.5)
        cls.treatment_leak_factor2 = cls.getp("treatment_leak_factor2", 0.0)
        
        cls.treatment_hood_size = cls.getp("treatment_hood_size", 3.0)
        
        cls.select_words = cls.getp("select_words", "qpu") 

        cls.assoc_train_method =  cls.getp("assoc_train_method","and")  # "none", "or", "and", "p", "always"

        cls.sessions_per_week = cls.getp("sessions_per_week", 2.0)

        ### TODO:
        cls.treatment_normal_exp_weeks = cls.getp("treatment_normal_exp_weeks", 0.5)
        cls.treatment_normal_exp_train_all = cls.getp("treatment_normal_exp_train_all", False)
        
        cls.leak_train_map_method =  cls.getp("leak_train_map_method", "none")  # "none", "sem", "cross", "both" 
        cls.leak_train_map_factor =  cls.getp("leak_train_map_factor", 0.0)

        cls.leak_use_fake_act = cls.getp("leak_use_fake_act", False)

        



    
################## HUMAN DATA #########################
        
class H:

    file_name = None
    frame = None
    hid = None

    human = None


    stroke = patient = None        
    eng_exp = None

    screener_eng = None
    screener_spa = None
    bnt_eng = None
    bnt_spa = None
    papt = None

    baseline_eng = None
    baseline_spa = None
    
    age_at_stroke = None
    age = None

    treatment_language = None
    treatment_sessions = None

    
    @classmethod
    def load(cls, fname):
        cls.frame = pickle.load(open(fname))
        cls.file_name = fname
        
    @classmethod
    def getp(cls, name, default=None):
        if name in cls.human:
            return cls.human[name]        
        elif name.upper() in cls.human:
            return cls.human[name.upper()]       
        elif name.lower() in cls.human:
            return cls.human[name.lower()]
        else:       
            return default

        
    @classmethod
    def eng_expf(cls, post=False):
        if not post:
            return lambda age: cls.eng_exp[int(floor(age))]
        else:
            return lambda age: cls.eng_exp[int(ceil(min(age, cls.age)))]

        
    @classmethod
    def set_hid(cls, hid):

        cls.hid = hid
        cls.human = cls.frame.ix[hid]
        
        cls.stroke = cls.patient = cls.getp("patient", False)
        cls.age = cls.getp("age")
        cls.age_at_stroke = cls.getp("age_at_stroke")

        cls.eng_exp = np.clip([cls.human["ENG_EXP%d"%i] for i in range(int(cls.age+2))], P.min_exp, 1.0 - P.min_exp)
        cls.spa_exp = 1.0 - cls.eng_exp

        cls.screener_eng = cls.getp("screener_en")
        cls.screener_spa = cls.getp("screener_spa")
        cls.bnt_eng = cls.getp("bnt_eng")
        cls.bnt_spa = cls.getp("bnt_spa")
        cls.papt = cls.getp("papt")

        cls.baseline_eng = cls.getp("baseline_eng")
        cls.baseline_spa = cls.getp("baseline_spa")
        
        cls.treatment_language = cls.getp("treatment_language", None)
        cls.treatment_sessions = cls.getp("ntreatment_sessions", 0)
        
        return hid

        
##################### TRAINING DATA ##########################

class W:

    nwords = None
    nsem_features = None
    neng_features = None
    nspa_features = None
    
    sem_data = None
    eng_data = None
    spa_data = None

    words = None

    rare_words = None
    common_words = None
    
    @classmethod
    def load(cls, fname):
        
        wdata = pickle.load(open(fname))
        cls.sem_data, cls.eng_data, cls.spa_data, cls.sem_count, cls.sem_yes, cls.categories, cls.papt, cls.papt2, cls.papt3, rare_words = wdata
        
        cls.words = cls.sem_data.index                   
        cls.nwords = cls.sem_data.shape[0]
        cls.nsem_features = cls.sem_data.shape[1]
        cls.neng_features = cls.eng_data.shape[1]
        cls.nspa_features = cls.spa_data.shape[1]

        cls.rare_words = rare_words.index
        cls.common_words = cls.sem_data.index[~cls.sem_data.index.isin(cls.rare_words)]

        cls.word_order = pd.Series(index=cls.words, data=np.arange(cls.nwords))
        
    @classmethod
    def word2int(cls, words):
        return cls.word_order[words].values
    
        
    @classmethod
    def word_freq(cls):
        word_freq = pd.Series(index=cls.words, data=np.ones(cls.nwords))
        word_freq[cls.rare_words] = np.linspace(1.0, P.min_word_freq, len(cls.rare_words))
        word_freq /= word_freq.sum()
        return word_freq
