import os,sys
import time
from math import sqrt, floor, ceil
import cPickle as pickle


import json

import numpy as np
import pandas as pd
import types

import tensorflow as tf
from lex4 import *

from params import P, H, W, T, init_params


hostname = os.popen("hostname").readline().strip()


print sys.argv


start_time = time.time()

opt = init_params(sys.argv[1:])
ID = "_".join(opt.ID)
logfile = open("%s.log"%ID, 'a')

Lex.logfile = logfile

res = []
details = []



################## LOG #########################

def log(msg, nl=True, flush=True, echo=True):

    logfile.write(str(msg))
    if echo:
        sys.stdout.write(str(msg))

    if nl:
        logfile.write("\n")
        if echo:
            sys.stdout.write("\n")

    if flush:
        logfile.flush()
        if echo:
            sys.stdout.flush()

#################  RANDOM SEEDS   ################


def set_random_seeds():

    if P.pre_seed is not None and P.pre_seed != -1:
        log("INFO:   pre-stroke seed found, %d"%(P.pre_seed))
    else:
        log("INFO:   creating pre-stroke seed...")
        P.pre_seed = int(os.urandom(4).encode('hex'), 16)


    if P.lesion_run:
        
        if P.sem_lesion_seed is not None and P.sem_lesion_seed != -1:
            log("INFO:   sem_lesion_seed found, %d"%(P.sem_lesion_seed))
        else:
            log("INFO:   creating sem_lesion_seed...")
            P.sem_lesion_seed = int(os.urandom(4).encode('hex'), 16)
        
        if P.eng_lesion_seed is not None and P.eng_lesion_seed != -1:
            log("INFO:   eng_lesion_seed found, %d"%(P.eng_lesion_seed))
        else:
            log("INFO:   creating eng_lesion_seed...")
            P.eng_lesion_seed = int(os.urandom(4).encode('hex'), 16)

        if P.spa_lesion_seed is not None and P.spa_lesion_seed != -1:
            log("INFO:   spa_lesion_seed found, %d"%(P.spa_lesion_seed))
        else:
            log("INFO:   creating eng_lesion_seed...")
            P.spa_lesion_seed = int(os.urandom(4).encode('hex'), 16)

        if P.post_seed is not None and P.post_seed != -1:
            log("INFO:   post_seed found, %d"%(P.post_seed))
        else:
            log("INFO:   creating post_seed...")
            P.post_seed = int(os.urandom(4).encode('hex'), 16)

    if P.treatment_run or ti >= 0:
        
        if T.treatment_seed is not None and T.treatment_seed != -1:
            log("INFO:   treatment_seed found, %d"%(T.treatment_seed))
        else:
            log("INFO:   creating treatment_seed...")
            T.treatment_seed = int(os.urandom(4).encode('hex'), 16)
            
        

                             
#######################################  MAIN  #################################

log("INFO: run ID %s"%ID)
log("INFO: start time %s"%time.asctime())
log("INFO: command line %s"%(" ".join(sys.argv)))
log("INFO: running on %s"%hostname)


### LOAD INPUT DATA

W.load(opt.words)
log("read %d words from %s"%(W.nwords, opt.words))

log("reading parameters from %s..."%opt.params)
P.load(opt.params)

if opt.treatments is not None:
    log("INFO: reading treatments from %s..."%opt.treatments)
    T.load(opt.treatments)
else:
    log("INFO: Not loading any treatments...")

    
log("INFO: reading humans from %s..."%opt.humans)
H.load(opt.humans)

### START TF SESSION    
config = tf.ConfigProto()         
if opt.nice:
    log("NOT hogging GPU memory")
    config.gpu_options.allow_growth=True
    config.gpu_options.visible_device_list = '1'
else:
    log("Hogging GPU memory")

sess = tf.Session(config=config)
lex = None


#############################################################################



with tf.device("/gpu:0"), sess.as_default():
    
    current_start_net = None
    ti = None
        
    #log("INFO: all are %s; %d pis, %d runs"%(hid, len(opt.pi), opt.nruns))
        
    if  opt.start_net is None:
        opt.start_net = [opt.start_net] * len(opt.hid)
    elif len(opt.start_net)==1:
        opt.start_net = opt.start_net * len(opt.hid)

                        
    log("INFO: start_nets: %r"%opt.start_net)
        

    ################ HID #################
    
    for iii, (hid, start_net) in enumerate(zip(opt.hid, opt.start_net)):

        log("INFO: starting hid = %r (may be None)"%hid)
        log("INFO: start_net = %r (may be None)"%start_net)


        ################### PI ###################
        
        for jjj,pi in enumerate(opt.pi):

            if opt.zip and iii != jjj:
                continue
            
            log("INFO:   starting pi = %d"%pi)
                        
            P.set_pi(pi)            

            if hid is None:
                if P.hid is not None:
                    log("INFO:   hid and start_net are from P...")
                    hid = P.hid
                    start_net = P.start_net
                else:
                    log("ERROR:   need hid from SOMEWHERE...")
                    sys.exit(1)

            log("INFO:   setting hid = %s"%hid)
            H.set_hid(hid)                       

            if H.stroke and P.treatment_run:
            
                log("INFO:       This is a treatment run")
                ti = opt.ti[jjj]
                log("INFO:       setting ti = %d"%(ti))
                T.set_ti(ti)

            else:
                log("INFO:       NOT a treatment run")
                ti = -1
               
            ### LOAD OR CREATE NET, IF NECESSARY
            log("INFO:   start_net = %r"%start_net) 
            
            if start_net is not None and not current_start_net == start_net:
                if not os.path.exists(start_net):
                    log("ERROR:     initial net %s not found"%start_net)
                    sys.exit()
                else:
                    log("INFO: about to load start_net from %s..."%start_net)
                    
                lex = Lex.load_npz(start_net, P, T, W, naming_only=opt.naming_only) 
                current_start_net = start_net

                log("INFO: loaded net, start age = %.2f"%lex.start_age)
                
                if lex.start_age == 0.0:
                    if P.start_age is not None:
                        log("WARNING:   initial net has no age; using start age %.2f from %s"%(P.start_age, opt.params))
                        lex.start_age = P.start_age
                    else:
                        log("ERROR:   need start age for initial net")
                        sys.exit()

                log("INFO: loaded net from %s; start age used: %.2f"%(start_net, lex.start_age))
                
            elif start_net is not None and current_start_net == start_net:
                log("INFO: looks like start net is already loaded; NOT loading it again...")
                
            elif lex is None or lex.x != P.map_size or lex.act_fun != P.act_fun:
                log("INFO:     starting fresh net, %s"%("naming only" if opt.naming_only else "full assoc"))
                lex = Lex(W.sem_data, W.eng_data, W.spa_data, x=P.map_size, y=P.map_size, naming_only=opt.naming_only, act_fun=P.act_fun)

            set_random_seeds()   #  the ones not provided by pfile, tfile, cmd line, or start_net  -Uli 7/17/18
                             
            tf.set_random_seed(P.pre_seed)
            np.random.seed(P.pre_seed)      

            
            lex.init_model(P, H, sess)
            log("INFO:    init/reset model done. age=%.2f"%lex.age)               

            
            ##################### RUN SIMULATION ######################

            # initial test, call it epoch -1, session -1
            if opt.initial_test:
                row,frame = lex.test_all(P, W, H, T, sess=sess, test_trans=True, epoch=-1, session=-1)
                res.append(row)        
                details.append(frame)

            end_age = H.age

            if H.stroke and not P.lesion_run:
                log("INFO:       H.stroke %r"%(H.stroke))
                end_age = H.age_at_stroke

            log("INFO:       training up to age %.2f..."%end_age)
            epoch = 0
            last_test_age = lex.age

            while(lex.age < end_age):

                t = time.time() 

                # how much are we training in this epoch, in 'years'?
                step = min(end_age-lex.age, 1.0)

                log("INFO:        about to decide whether to lesion")
                log("INFO:        lesion_run %r, stroke %r, lex.lesioned %r"%(P.lesion_run, H.stroke, lex.lesioned) )
                

                if P.lesion_run and H.stroke and not lex.lesioned:
                    if lex.age >= H.age_at_stroke:

                        ################  LESION  ##############

                        log("INFO:       applying lesions at age = %.2f"%lex.age)
                        log("INFO:         sem %s %r"%(P.semantic_lesion_type, P.semantic_lesion_strength))
                        log("INFO:         eng %s %r"%(P.phonetic_lesion_type, P.eng_lesion_strength))
                        log("INFO:         spa %s %r"%(P.phonetic_lesion_type, P.spa_lesion_strength))

                        lex.apply_lesion(P, H, W, T, sess=sess)

                        ### after lesion, set post-lesion training seed (up to treatment)
                        tf.set_random_seed(P.post_seed)
                        np.random.seed(P.post_seed)              

                        ### make sure we test right after lesion
                        row,frame = lex.test_all(P, W, H, T, sess=sess, test_trans=True, epoch=epoch, session=-1)
                        res.append(row)        
                        details.append(frame)

                        if opt.save_post:
                            log("INFO:       saving post_stroke net...", nl=False)
                            t = time.time()
                            if opt.save_prefix is not None:
                                fname = "%s_%s_%d"%(opt.save_prefix, hid, pi)
                            else:
                                fname = "%s_%s_%d"%(ID, hid, pi)                   
                            if opt.run is not None:
                                fname += "_%d"%opt.run
                            fname += ".post_stroke.net"                
                            lex.save_npz(fname, P, T, sess=sess)      
                            log("INFO:       saved as %s.npz; took %.3fs\n"%(fname, time.time()-t))



                    else:
                        ### we're lesioning, but it's not time yet. Don't train too far though
                        log("INFO:        are we lesioning? ... %s"%("YES but not now" if P.lesion_run else "NO"))
                        step = min(H.age_at_stroke-lex.age, 1.0)

                log("INFO:       step = %.4f"%step)

                #### ...TRAIN!   ###
                lex.noise_assoc(P, fraction=step) # noise less if we're training for less than a year

                lex.train(lex.age+step, P, H, W, sess)
                lex.norm_assoc(P, sess)

                log("INFO:      epoch %d took %.3fs, age now %.3f"%(epoch, time.time()-t, lex.age))
                log("INFO:      %d patterns, %d eng, %d spa"%(lex.npatterns, lex.neng_patterns, lex.nspa_patterns))
                log("INFO:      epoch %d took %.3fs, age now %.3f"%(epoch, time.time()-t, lex.age))

                
                if lex.age >= H.age_at_stroke and not lex.lesioned and opt.save_pre:
                    log("INFO:       saving pre_stroke net...", nl=False)
                    t = time.time()
                    if opt.save_prefix is not None:
                        fname = "%s_%s_%d"%(opt.save_prefix, hid, pi)
                    else:
                        fname = "%s_%s_%d"%(ID, hid, pi)                   
                    if opt.run is not None:
                        fname += "_%d"%opt.run
                    fname += ".pre_stroke.net"                
                    lex.save_npz(fname, P, T, sess=sess)      
                    log("INFO:       saved as %s.npz; took %.3fs\n"%(fname, time.time()-t))


                    
                if not P.treatment_run:
                    epoch += 1

                    if (((lex.age - last_test_age) >= opt.test_freq) or lex.age >= end_age):
                        log("INFO:        testing...")
                        row,frame = lex.test_all(P, W, H, T, sess=sess, test_trans=True, epoch=epoch, session=-1)
                        res.append(row)        
                        details.append(frame)

                        last_test_age = lex.age


            ############### TREATMENT #################

            if opt.save_pre_treat:
                log("INFO:       saving pre_treat net...", nl=False)
                t = time.time()
                if opt.save_prefix is not None:
                    fname = "%s_%s_%d"%(opt.save_prefix, hid, pi)
                else:
                    fname = "%s_%s_%d"%(ID, hid, pi)                   
                if opt.run is not None:
                    fname += "_%d"%opt.run
                fname += ".pre_treat.net"                
                lex.save_npz(fname, P, T, sess=sess)      
                log("INFO:       saved as %s.npz; took %.3fs\n"%(fname, time.time()-t))


            if ti is not None and ti != -1:

                log("INFO:       TREATING...")


                log("INFO:       assessment at %.2f, %d sessions"%(H.age, H.treatment_sessions))
                
                ########### TREATMENT SEED

                if T.treatment_seed is not None and T.treatment_seed != -1:
                    log("INFO:   T.treatment_seed found, %d"%(T.treatment_seed))
                else:
                    log("INFO:   creating T.treatment_seed...")
                    T.treatment_seed = int(os.urandom(4).encode('hex'), 16)

                tf.set_random_seed(T.treatment_seed)
                np.random.seed(T.treatment_seed)             

                if H.treatment_sessions is None or H.treatment_sessions <= 0:
                    log("INFO:    looks like we're missing a sessions count... assuming 20")
                    H.treatment_sessions = 20

                if H.treatment_language is None or H.treatment_language not in ['eng', 'spa']:
                    log("INFO:    looks like there's no treatment language in humans file...")
                    H.treatment_language = opt.treatment_language                       
                    log("INFO:    cmd line says %r"%opt.treatment_language)

                elif opt.flip_language:
                    real_lang = H.treatment_language
                    H.treatment_language = 'eng' if real_lang == 'spa' else 'spa'
                    log("INFO:    FLIPPING TREATMENT LANGUAGE!!! %s -> %s"%(real_lang, H.treatment_language))



                log("INFO:       STARTING TREATMENT, age = %.3f"%(lex.age))    
                lex.start_treatment(P, H, W, T, sess)

                epoch += 1

                row,frame = lex.test_all(P, W, H, T, sess=sess, test_trans=True, epoch=epoch, session=0)
                res.append(row)        
                details.append(frame)


                for treatment_session in range(H.treatment_sessions):
                    epoch += 1

                    log("INFO:      TREATING")
                    lex.treat_one_session(P, H, W, T, sess)
                    lex.norm_assoc(P, sess)  ### added 7/2/18 -uli

                    log("INFO:      TESTING")
                    row,frame = lex.test_all(P, W, H, T, sess=sess, test_trans=True, epoch=epoch, session=treatment_session+1)
                    res.append(row)        
                    details.append(frame)


                if opt.save_post_treat:

                    log("INFO:       saving post_treat net...", nl=False)
                    t = time.time()
                    if opt.save_prefix is not None:
                        fname = "%s_%s_%d_%d"%(opt.save_prefix, hid, pi, ti)
                    else:
                        fname = "%s_%s_%d_%d"%(ID, hid, pi, ti)                   
                    if opt.run is not None:
                        fname += "_%d"%opt.run
                    fname += ".post_treat.net"                
                    lex.save_npz(fname, P, T, sess=sess)      
                    log("INFO:       saved as %s.npz; took %.3fs\n"%(fname, time.time()-t))

            log("INFO:    pi %d done."%pi)
            
        log("INFO:   hid %s done."%hid)
        
    log("INFO: done!")
    sess.close()
    
    log("INFO: Saving results as %s.res.csv"%ID)                   
    result_frame = pd.DataFrame(res)

    result_frame['ID'] = ID
    result_frame['run'] = opt.run
        
    result_frame['date'] = time.asctime()   
    result_frame['run_sec'] = time.time()-start_time
    
    result_frame['hfile'] = H.file_name
    result_frame['pfile'] = P.file_name
    result_frame['tfile'] = T.file_name

                       
    result_frame.set_index(["ID", "hid", "pi", "ti", "epoch", "session", "treat_lang", "run"]).to_csv("%s.res.csv"%ID, header=True)
    
    if opt.save_outputs:
        log("INFO: Saving details as %s.details.pkl"%ID)          
        pd.concat(details).to_pickle("%s.details.pkl"%ID)

    log("INFO: all done! %s\nKBAI"%time.asctime())
    logfile.close()
