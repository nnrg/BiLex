
from math import sqrt
import sys, time, os, types

import cPickle as pickle
import numpy as np
import pandas as pd

import tensorflow as tf


################################# LESION HELPERS ###############################

class Lexception(Exception):
    pass


def lesion_mask_flat(shape, p):

    mask = np.floor(np.random.random(shape) + p)
    
    if np.sum(mask) == shape[0]*shape[1]:
        mask[0,0] = 0.0
        
    return np.array(mask, dtype=np.float32)


def lesion_mask_gauss(shape, strength):

    sigma = float(shape[0] * strength)
    
    if sigma==0.0:
        return np.zeros(shape, dtype=np.float32) 

    x = np.square(np.arange(shape[0])-np.floor(shape[0]/2))
    y = np.square(np.arange(shape[1])-np.floor(shape[1]/2))
    p = np.exp((x.reshape([-1,1]) + y.reshape([1,-1]))/(-sigma*sigma))

    mask = np.floor(np.random.random(shape) + p) 

    if np.sum(mask) == shape[0]*shape[1]:
        mask[0,0] = 0.0
    
    return np.array(mask, dtype=np.float32) 


def lesion_mask_round(shape, strength):

    if strength==0.0:
        return np.zeros(shape, dtype=np.float32) 
    
    R = strength * np.sqrt(shape[0]*shape[0]+shape[1]*shape[1]) / 2.0

    if R <= shape[0]/2.0:
        centerx = np.random.randint(R-1, shape[0]-R+1)
    else:
        centerx = shape[0]/2
        
    if R <= shape[1]/2.0:        
        centery = np.random.randint(R-1, shape[1]-R+1)
    else:
        centery = shape[1]/2
            
    x = np.square(np.arange(shape[0])-centerx)   #-np.floor(shape[1]/2))
    y = np.square(np.arange(shape[1])-centery)#-np.floor(shape[1]/2))
    p = np.array(np.sqrt(x.reshape([-1,1]) + y.reshape([1,-1])) < R, dtype=np.float32)

    mask = np.floor(p)

    if np.sum(mask) == shape[0]*shape[1]:
        mask[0,0] = 0.0
    
    return np.array(mask, dtype=np.float32) 


############################## RAW TENSORFLOW SOM ##############################


class som_:
    
    def __init__(self, x, y, dim, masked=True,
                 W=None, input_=None, exp_mask_=None,
                 sigma_=None, alpha_=None, R_=None,
                 maxR=None, minR=None, act_fun='gauss', name="SOM"):

        self.name = name
        self.act_fun = act_fun
        
        self.dtype = tf.float32    
        self.itype = tf.int32  
        
        with tf.name_scope(self.name):
                    
            ### MODEL PARAMS                        
            if W is None:
                self.x, self.y, self.dim = x, y, dim
                self.W_ = tf.Variable(tf.random_uniform([self.x,self.y,self.dim], minval=0.0, maxval=0.1, dtype=self.dtype), 
                                      dtype=self.dtype, 
                                      name="W")
            else:
                self.x, self.y, self.dim = W.shape
                self.W_ = tf.Variable(W, 
                                      self.dtype, 
                                      name="W")   
            
            zeroi_ = tf.constant(0, self.itype)
            onei_ = tf.constant(1, self.itype)
            zerof_ = tf.constant(0., self.dtype)
            onef_ = tf.constant(1., self.dtype)
            
            self.xi_ = tf.constant(self.x, self.itype)
            self.yi_ = tf.constant(self.y, self.itype)
            self.xf_ = tf.constant(self.x, self.dtype)
            self.yf_ = tf.constant(self.y, self.dtype)

            if masked:
                self.mask_ = tf.placeholder(shape=[None,None], dtype=self.dtype)
            else:
                self.mask = tf.constant([[0.0]], dtype=self.dtype)

            self.large_dist = tf.constant(self.dim, dtype=self.dtype)
                
            ### TRAINING PARAMETERS  
            with tf.name_scope("training_params"):
                                   
                self.input_ = tf.placeholder(dtype=self.dtype, shape=[None, self.dim], name="input") if input_ is None else input_ 
                self.nbatch = tf.shape(self.input_, out_type=self.itype)[0]
                self.nbatch_list_ = tf.range(self.nbatch, dtype=self.itype)
                                
                self.alpha_ = alpha_ if alpha_ is not None else tf.placeholder(shape=[], dtype=self.dtype, name="alpha")                    
                self.sigma_ = sigma_ if sigma_ is not None else tf.placeholder(shape=[], dtype=self.dtype, name="sigma")                                          
                self.R_ = R_ if R_ is not None else tf.placeholder(shape=[], dtype=self.dtype, name="R")
        
            ### PRESENT
            with tf.name_scope("present"):
                    
                    self.raw_diff_ = tf.subtract(tf.reshape(self.input_,
                                                            [-1,1,1,self.dim],
                                                            name="reshape_input"),
                                                 tf.reshape(self.W_,
                                                            [1, self.x,self.y,self.dim]))
    
        
                    self.dist2_ = tf.reduce_sum(tf.square(self.raw_diff_),axis=3)
                                                                      
                    
                    self.winner_ = tf.argmin(tf.reshape(self.dist2_ + tf.multiply(tf.expand_dims(self.mask_, 0),
                                                                                  self.large_dist),
                                                        [-1, self.x*self.y]),
                                             axis=1,
                                             name="winner",
                                             output_type=self.itype) 
                                           
                    
                    self.wx_ = tf.floordiv(self.winner_, self.yi_, name="wx") #tf.scatter_nd(self.exp_indices_, winner_ // self.y, [self.nmaps])            
                    self.wy_ = tf.floormod(self.winner_, self.yi_, name="wy")


                    
                    ### NEW-STYLE LABELS -- note lesion mask is not applied (don't care that units that don't exist have labels) 
                    
                    self.labels_ = tf.subtract(tf.argmin(self.dist2_,
                                                         axis=0,
                                                         name="labels",
                                                         output_type=self.itype),
                                               tf.multiply(99999,
                                                           tf.cast(self.mask_, self.itype)))
                       
            ### PAPT - this may only be useful for previous sim?
            self.papt_cues_ = tf.placeholder(dtype=self.itype, shape=[None], name="cues")
            self.papt_guess_ = tf.gather_nd(self.W_, tf.stack([self.wx_, self.wy_, self.papt_cues_], axis=-1))

                                    
            ### BUILD SPARSE INDICES
            with tf.name_scope("indices"):

                self.circlef_ = self.make_circle()    
                self.circlei_ = tf.cast(self.circlef_, self.itype)                                  

                self.winner_indices_ = tf.reshape(tf.stack([self.nbatch_list_, self.wx_, self.wy_],
                                                            axis=-1),
                                                  [-1,1,3])
                                                                 
                self.indices_act_ = tf.add(self.winner_indices_, self.circlei_)

                self.indices_W_ = self.indices_act_[:,:,1:]
                
                
                

    
            with tf.name_scope("train"):
                    
                self.sparse_theta_ = tf.exp(tf.divide(tf.norm(self.circlef_,
                                                              axis=-1),
                                                      -2.0*tf.square(self.sigma_)),
                                            name="sparse_theta")
                                                               
                self.sparse_alpha_theta_ = self.alpha_ * self.sparse_theta_
                
                self.delta_ = tf.multiply(tf.expand_dims(self.sparse_alpha_theta_,-1),
                                          tf.gather_nd(self.raw_diff_,
                                                       self.indices_act_))
                                                       
                self.train_ = tf.scatter_nd_add(self.W_,
                                                self.indices_W_,
                                                self.delta_,
                                                use_locking=None,
                                                name="scatter_W_updates")  
        

            with tf.name_scope("act"):                   

                self.sparse_mask_ = tf.gather_nd(self.mask_, self.indices_W_)
                
                self.sparse_dist_ = tf.sqrt(tf.gather_nd(self.dist2_, 
                                                         self.indices_act_))


                
                self.min_ = tf.sqrt(tf.gather_nd(self.dist2_, self.winner_indices_))

                self.max_ = tf.maximum(self.min_ + 1.0,
                                       tf.reduce_max(tf.multiply(self.sparse_dist_,
                                                     1.0-self.sparse_mask_),
                                                     axis=1,
                                                     keep_dims=True))
                                 
                self.sparse_classic_act_ = tf.multiply((self.max_ - self.sparse_dist_) / (self.max_-self.min_),
                                                       1.0-self.sparse_mask_)
                
                self.classic_act_ = tf.scatter_nd(self.indices_act_, self.sparse_classic_act_, [self.nbatch, self.x, self.y])
                
                self.sparse_gauss_act_ = tf.multiply(self.sparse_classic_act_, self.sparse_theta_)   
                self.gauss_act_ = tf.scatter_nd(self.indices_act_, self.sparse_gauss_act_, [self.nbatch, self.x, self.y])

                
                if self.act_fun == 'classic':
                    self.sparse_act_ = self.sparse_classic_act_
                    self.act_ = self.classic_act_

                elif self.act_fun == 'gauss':
                    self.sparse_act_ = self.sparse_gauss_act_
                    self.act_ = self.gauss_act_
                else:
                    raise ValueError("unknown act_fun: " + self.act_fun)  


            
        ##################### LESIONS ########################
        
        self.lesion_strength_ = tf.placeholder(shape=[], dtype=self.dtype, name="sem_lesion")
        self.lesion_noise_ = tf.assign_add(self.W_,
                                                    tf.random_uniform([self.x, self.y, self.dim],
                                                                      minval=-self.lesion_strength_/2.0,
                                                                      maxval=self.lesion_strength_/2.0,
                                                                      dtype=self.dtype),
                                                name="semantic_lesion_noise")

        w_mean_, w_var_ = tf.nn.moments(self.W_, axes=[0,1]) #keep_dims=True
        w_std_ = tf.sqrt(w_var_)
                                                      
        self.lesion_noise2_ = tf.assign(self.W_,
                                                 tf.maximum(0.0,
                                                            tf.add(self.W_,
                                                                   tf.add(w_mean_,
                                                                          tf.multiply(w_var_,
                                                                                      tf.random_normal([self.x, self.y, self.dim],
                                                                                      mean=0.0,
                                                                                      stddev=self.lesion_strength_,
                                                                                      dtype=self.dtype))))),
                                                name="semantic_lesion_noise2")

        self.lesion_noise3_ = tf.assign(self.W_,
                                        tf.maximum(0.0,
                                                   tf.add(self.W_,
                                                          tf.multiply(w_std_,
                                                                      tf.random_normal([self.x, self.y, self.dim],
                                                                                       mean=0.0,
                                                                                       stddev=self.lesion_strength_,
                                                                                       dtype=self.dtype)))),
                                        name="semantic_lesion_noise3")


        # (previous lesion, not used)
        # random_binary_W_ = tf.floor(1.0 + tf.random_uniform([self.x, self.y, self.dim], dtype=self.dtype) - self.lesion_strength_)    
        #                                                           
        #self.lesion_prune_ = tf.assign(self.W_,
        #                                                tf.multiply(self.W_,
        #                                                            random_binary_W_),
        #                                                name="prune_sem")



                            
    def make_circle(self):
            
        square_ = tf.stack(tf.meshgrid(tf.zeros([1], dtype=self.dtype),
                                       tf.range(tf.floor(-self.R_), tf.ceil(self.R_+1.0), 1.0, dtype=self.dtype),
                                       tf.range(tf.floor(-self.R_), tf.ceil(self.R_+1.0), 1.0, dtype=self.dtype),
                                       indexing='ij'),
                           axis=-1)       
        
        circle_ = tf.reshape(tf.gather_nd(square_,
                                          tf.where(tf.less_equal(tf.norm(square_, axis=-1),
                                                                 self.R_))),
                                  [1,-1,3])
        
        return circle_




############################## RAW TENSORFLOW LEXICON ##############################

class lex_:
    
    
    def __init__(self, sem_data, eng_data, spa_data, x=None, y=None, naming_only=False,
                 act_fun='gauss', name="DISCERN"):
        
        self.name = name
        self.dtype = tf.float32
        self.itype = tf.int32
        
        self.act_fun = str(act_fun)


        
        self.i_ = tf.placeholder(shape=([None]), dtype=self.itype, name="i")
        self.nbatch = tf.size(self.i_, out_type=self.itype)
        
        self.map_alpha_ = tf.placeholder(shape=[], dtype=self.dtype, name="map_alpha")
        self.alpha_ = tf.placeholder(shape=[], dtype=self.dtype, name="alpha")
        self.sigma_= tf.placeholder(shape=[], dtype=self.dtype, name="sigma")
        self.R_ = tf.placeholder(shape=[], dtype=self.dtype, name="R")
        
        self.noise_std_ = tf.placeholder(shape=[], dtype=self.dtype, name="noise_var")

        self.leak_alpha_ = tf.placeholder(shape=[], dtype=self.dtype, name="leak_alpha")
        self.leak_alpha2_ = tf.placeholder(shape=[], dtype=self.dtype, name="leak_alpha2")
        
        self.ninp = len(sem_data)
        
        self.sem_data = sem_data
        self.eng_data = eng_data
        self.spa_data = spa_data
        
        self.sem_data_ = tf.constant(sem_data.values, dtype=self.dtype)
        self.eng_data_ = tf.constant(eng_data.values, dtype=self.dtype)
        self.spa_data_ = tf.constant(spa_data.values, dtype=self.dtype)

        self.x, self.y  = x,y
    
        self.sem = som_(x,y,sem_data.shape[1], input_=tf.gather(self.sem_data_, self.i_), 
                         sigma_=self.sigma_, alpha_=self.map_alpha_, R_=self.R_, 
                         act_fun=self.act_fun, name=self.name+"_sem")
        self.eng = som_(x,y,eng_data.shape[1], input_=tf.gather(self.eng_data_, self.i_), 
                        sigma_=self.sigma_, alpha_=self.map_alpha_, R_=self.R_,
                        act_fun=self.act_fun, name=self.name+"_eng")
        self.spa = som_(x,y,spa_data.shape[1], input_=tf.gather(self.spa_data_, self.i_),
                        sigma_=self.sigma_, alpha_=self.map_alpha_, R_=self.R_,
                        act_fun=self.act_fun, name=self.name+"_spa")        

        
        if naming_only:
            self.naming_only = True
            self.nassoc = 2
            self.SEM2ENG = 0
            self.SEM2SPA = 1

            self.src_map = [self.sem, self.sem]
            self.tgt_map = [self.eng, self.spa]
               
        else:
            self.naming_only = False
            self.nassoc = 6
            self.SEM2ENG = 0
            self.ENG2SEM = 1   
            self.SEM2SPA = 2
            self.SPA2SEM = 3 
            self.ENG2SPA = 4       
            self.SPA2ENG = 5

            self.src_map = [self.sem, self.eng, self.sem, self.spa, self.eng, self.spa]
            self.tgt_map = [self.eng, self.sem, self.spa, self.sem, self.spa, self.eng]         

        
        normval = np.float32(1/sqrt(self.x*self.y))
        self.assoc_ = tf.Variable(tf.ones([self.nassoc, x, y, x, y]) * normval,
                                      self.dtype, 
                                      name=self.name+"_assoc%d"%self.nassoc)

        



            

        
        ### TRAINING OPS
        self.train_maps_only = [self.sem.train_, self.eng.train_, self.spa.train_] 
                           
        ### PROP OPS
        
        if self.naming_only: ### NAMING ONLY OPS
            
            self.prop_act_ = [self.make_prop_act(self.SEM2ENG), self.make_prop_act(self.SEM2SPA)]
            wx_e,wy_e = self.make_prop_winner(self.SEM2ENG)
            wx_s,wy_s = self.make_prop_winner(self.SEM2SPA)
            self.prop_wx_ = [wx_e, wx_s]
            self.prop_wy_ = [wy_e, wy_s]

            
            i12e,i21e,dwe = self.assoc_deltas(self.SEM2ENG)
            i12s,i21s,dws = self.assoc_deltas(self.SEM2SPA)
            self.assoc_indices_ = [i12e,i12s]
            self.assoc_updates_ = [dwe,dws]

            self.train_english = [self.sem.train_, self.eng.train_, 
                                  self.make_assoc_train_op([self.SEM2ENG])]  

            self.train_spanish = [self.sem.train_, self.spa.train_, 
                                  self.make_assoc_train_op([self.SEM2SPA])] 

            self.train_translation = [] 

            self.train_all     = [self.sem.train_, self.eng.train_, self.spa.train_, 
                                  self.make_assoc_train_op([self.SEM2ENG, self.SEM2SPA])]
   
            
        else:   ### FULL ASSOC OPS
            self.prop_act_ = []
            self.prop_wx_ = []
            self.prop_wy_ = []
            self.prop_winner_vectors_ = []

            for i in range(6):
                self.prop_act_.append(self.make_prop_act(i))
                wx,wy = self.make_prop_winner(i)
                self.prop_wx_.append(wx)
                self.prop_wy_.append(wy)
                self.prop_winner_vectors_.append(tf.gather_nd(self.tgt_map[i].W_, tf.stack([wx,wy], axis=-1)))

                
            self.assoc_updates_ = []
            self.assoc_indices_ = []
            for i in range(0,6,2):
                i12,i21,dw = self.assoc_deltas(i)
                self.assoc_indices_ += [i12, i21]
                self.assoc_updates_ += [dw,dw]

            self.train_english = [self.sem.train_, self.eng.train_, 
                                  self.make_assoc_train_op([self.SEM2ENG, self.ENG2SEM])]  

            self.train_spanish = [self.sem.train_, self.spa.train_, 
                                  self.make_assoc_train_op([self.SEM2SPA, self.SPA2SEM])] 

            self.train_english_p = [self.sem.train_, self.eng.train_, 
                                          self.make_assoc_train_op([self.SEM2ENG, self.ENG2SEM, self.SPA2ENG])]  

            self.train_spanish_p = [self.sem.train_, self.spa.train_, 
                                          self.make_assoc_train_op([self.SEM2SPA, self.SPA2SEM, self.ENG2SPA])] 

            self.train_english_q = [self.sem.train_, self.eng.train_, 
                                          self.make_assoc_train_op([self.SEM2ENG, self.ENG2SEM, self.ENG2SPA])]  

            self.train_spanish_q = [self.sem.train_, self.spa.train_, 
                                          self.make_assoc_train_op([self.SEM2SPA, self.SPA2SEM, self.SPA2ENG])] 
            
            self.train_english_or = [self.sem.train_, self.eng.train_, 
                                          self.make_assoc_train_op([self.SEM2ENG, self.ENG2SEM, self.SPA2ENG, self.ENG2SPA])]  

            self.train_spanish_or = [self.sem.train_, self.spa.train_, 
                                          self.make_assoc_train_op([self.SEM2SPA, self.SPA2SEM, self.ENG2SPA, self.SPA2ENG])] 
            
            
            self.train_all     = [self.sem.train_, self.eng.train_, self.spa.train_, 
                                  self.make_assoc_train_op([self.SEM2ENG, self.ENG2SEM,
                                                            self.SEM2SPA, self.SPA2SEM,
                                                            self.ENG2SPA, self.SPA2ENG])]           

            ### leak means training sem <--> eng/spa connections using act leaked from other phonetic map 
            self.leak_train_eng_, self.leak_act_eng_ = self.make_leak_train_op(lang='eng')
            self.leak_train_spa_, self.leak_act_spa_ = self.make_leak_train_op(lang='spa') 


            ### leak 2 means training eng <--> spa connections using act leaked from semantic map 
            self.leak_train_eng2_, self.leak_act_eng2_ = self.make_leak_train_op2(lang='eng')
            self.leak_train_spa2_, self.leak_act_spa2_ = self.make_leak_train_op2(lang='spa')


            
        ###################### NORM ASSOC WEIGHTS ####################
            
        self.norm_assoc_classic_ = tf.assign(self.assoc_,
                                             tf.clip_by_norm(self.assoc_, 1.0, axes=[3,4], name="norm_assoc_classic"))       

        if self.naming_only:
            self.norm_assoc_combined_ = tf.assign(self.assoc_,
                                                  tf.clip_by_norm(self.assoc_, 2.0, axes=[0,3,4], name="norm_assoc_comb"))       
        else:
            sqsum = tf.reduce_sum(tf.square(self.assoc_), axis=[3,4], keep_dims=True)
            sem_norm = 2.0*tf.minimum(tf.sqrt(tf.add(sqsum[0], sqsum[2])), 2.0)
            eng_norm = 2.0*tf.minimum(tf.sqrt(tf.add(sqsum[1], sqsum[4])), 2.0)
            spa_norm = 2.0*tf.minimum(tf.sqrt(tf.add(sqsum[3], sqsum[5])), 2.0)

            self.norm_assoc_combined_ = tf.assign(self.assoc_,
                                                tf.divide(self.assoc_, tf.stack([sem_norm,
                                                                                 eng_norm,
                                                                                 sem_norm,
                                                                                 spa_norm,
                                                                                 eng_norm,
                                                                                 spa_norm],
                                                                                axis=0)),
                                                name="norm_assoc_combined")       
        

        self.norm_assoc_global_ = tf.assign(self.assoc_,
                                            tf.clip_by_norm(self.assoc_, self.x*self.y, axes=[1,2,3,4], name="norm_assoc_comb"))
        
            
        ############# LESIONS/DAMAGE #####################
        
        self.noise_assoc_normal_ = tf.assign_add(self.assoc_,
                                                 tf.random_normal([self.nassoc,self.x,self.y,self.x,self.y],
                                                                  mean=0.0,
                                                                  stddev=self.noise_std_,
                                                                  dtype=self.dtype),
                                                 name = "noise_assoc_normal")

        self.noise_assoc_uniform_ = tf.assign_add(self.assoc_,
                                                  tf.random_uniform([self.nassoc, self.x,self.y,self.x,self.y], minval=0.0, maxval=self.noise_std_, dtype=self.dtype),
                                                  name = "noise_assoc_uniform")  


        self.assoc_lesion_noise_ = self.noise_assoc_uniform_
        self.assoc_lesion_strength_ = self.noise_std_
        
    ## SEM2ENG = 0
    ## ENG2SEM = 1   
    ## SEM2SPA = 2
    ## SPA2SEM = 3 
    ## ENG2SPA = 4       
    ## SPA2ENG = 5
        
                                                                                                                                   
    def make_prop_act(self, assoc_i):

        if not self.naming_only:       
            soms = [self.sem, self.eng, self.sem, self.spa, self.eng, self.spa]
        else:
            soms = [self.sem, self.sem] ### src map for naming is always sem
            
        tgt_i = assoc_i + 1 - 2*(assoc_i%2)
        mask = tf.expand_dims(1.0 - soms[tgt_i].mask_, 0)
               
        return tf.multiply(tf.reduce_sum(tf.multiply(tf.reshape(soms[assoc_i].sparse_act_,
                                                                [self.nbatch, -1, 1 , 1]),
                                                     tf.gather_nd(self.assoc_[assoc_i],
                                                                  soms[assoc_i].indices_W_)),
                                         axis=1),
                           mask)


        
    def make_prop_winner(self, assoc_i):        
        act_ = self.prop_act_[assoc_i]
        winner_ = tf.argmax(tf.reshape(act_, [-1, self.x*self.y]), axis=1, output_type=self.itype)
        
        wx_ = tf.floordiv(winner_, self.y)      
        wy_ = tf.floormod(winner_, self.y)
        
        return wx_,wy_
    
    
    def make_assoc_train_op(self, assoc_is):        
        indices_ = tf.bitcast(tf.concat([tf.bitcast(self.assoc_indices_[i], tf.float32) for i in assoc_is] , 1) , self.itype)
        updates_ = tf.concat([self.assoc_updates_[i] for i in assoc_is] , 1)
        
        return tf.scatter_nd_add(self.assoc_, indices_, updates_, use_locking=False)   
    
        
    def assoc_deltas(self, assoc_i):

        if not self.naming_only:       
            som1s =[self.sem, self.eng, self.sem, self.spa, self.eng, self.spa]
            som2s =[self.eng, self.sem, self.spa, self.sem, self.spa, self.eng]
        else:
            som1s = [self.sem, self.sem] ### src map for naming is always sem
            som2s = [self.eng, self.spa] ### tgt map for naming is always phonetic
        
        som1 = som1s[assoc_i]
        som2 = som2s[assoc_i]
        
        act1_ = som1.sparse_act_
        act2_ = som2.sparse_act_
        
        deltas_ = self.alpha_ * tf.multiply(tf.expand_dims(act1_, -1),
                                        tf.expand_dims(act2_, -2))        
        
        indices12_ = tf.cast(tf.add(tf.constant([[[[assoc_i,0,0,0,0]]]], dtype=self.dtype),
                                    tf.add(tf.pad(tf.expand_dims(tf.cast(som1.indices_W_, self.dtype),
                                                                 -2),
                                                  [[0,0],[0,0],[0,0], [1,2]]),
                                           tf.pad(tf.expand_dims(tf.cast(som2.indices_W_, self.dtype),
                                                                 -3),
                                                  [[0,0],[0,0],[0,0], [3,0]]))),
                            self.itype)
        
        indices21_ = tf.cast(tf.add(tf.constant([[[[assoc_i+1,0,0,0,0]]]], dtype=self.dtype),
                                    tf.add(tf.pad(tf.expand_dims(tf.cast(som2.indices_W_, self.dtype),
                                                                -2),
                                                  [[0,0],[0,0],[0,0], [1,2]]),
                                           tf.pad(tf.expand_dims(tf.cast(som1.indices_W_, self.dtype),
                                                                 -3),
                                                  [[0,0],[0,0],[0,0], [3,0]]))),
                             self.itype)
               
        return indices12_, indices21_, deltas_



    def make_sparse_leak_act(self, lang='eng'):

        if self.naming_only:
            raise Lexception("Leaking works only with full connections")

        if lang == 'eng':
            som = self.eng
            act_ = self.prop_act_[self.SPA2ENG]
            wx_ = self.prop_wx_[self.SPA2ENG]
            wy_ = self.prop_wy_[self.SPA2ENG] 
        elif lang == 'spa':
            som = self.spa
            act_ = self.prop_act_[self.ENG2SPA]
            wx_ = self.prop_wx_[self.ENG2SPA]
            wy_ = self.prop_wy_[self.ENG2SPA] 


        leak_winner_indices_ = tf.reshape(tf.stack([som.nbatch_list_, wx_,wy_], axis=-1), [-1,1,3])
        leak_indices_act_ = tf.add(leak_winner_indices_, som.circlei_)
        leak_indices_W_ = leak_indices_act_[:,:,1:]

        raw_sparse_act_ = tf.multiply(tf.gather_nd(act_, leak_indices_act_), 1.0-som.sparse_mask_)

        
        max_ = tf.gather_nd(act_, leak_winner_indices_)

        sparse_classic_leak_act_ = tf.multiply(raw_sparse_act_ / max_,   #### applying classic norm first, 7/2/18 -uli
                                               1.0-som.sparse_mask_)     #### for now, just divide by the max... see if it fixed leak nans 7/8/28 -U   (yes)

        sparse_gauss_leak_act_ = tf.multiply(sparse_classic_leak_act_, som.sparse_theta_)   #### applying gauss act, 6/30/18 -uli   (possibly try without at some point... 7/2/18 -U)

        return leak_indices_act_, leak_indices_W_, sparse_gauss_leak_act_



        

    def make_leak_train_op(self, lang='eng'):

        if self.naming_only:
            raise Lexception("Leaking works only with full connections")

        if lang=='eng':
            som2 = self.eng
            assoc_i = self.SEM2ENG
        else:
            som2 = self.spa
            assoc_i = self.SEM2SPA

        som1 = self.sem
        act1_ = som1.sparse_act_

        leak_indices_act_, leak_indices_W_, sparse_leak_act_ =  self.make_sparse_leak_act(lang)

        deltas_ = self.leak_alpha_ * tf.multiply(tf.expand_dims(act1_, -1),
                                        tf.expand_dims(sparse_leak_act_, -2))        

        indices12_ = tf.cast(tf.add(tf.constant([[[[assoc_i,0,0,0,0]]]], dtype=self.dtype),
                                    tf.add(tf.pad(tf.expand_dims(tf.cast(som1.indices_W_, self.dtype),
                                                                 -2),
                                                  [[0,0],[0,0],[0,0], [1,2]]),
                                           tf.pad(tf.expand_dims(tf.cast(leak_indices_W_, self.dtype),
                                                                 -3),
                                                  [[0,0],[0,0],[0,0], [3,0]]))),
                            self.itype)

        ## Note: not bothering with phonetic -> semantic deltas... just because we never care about those connections.
        ## really should concatenate 12 and 21 indices and duplicate deltas of course... meh  4/29/18, Uli
        leak_train_ = tf.scatter_nd_add(self.assoc_, indices12_, deltas_, use_locking=False)   

        leak_act_ = tf.scatter_nd(leak_indices_act_, sparse_leak_act_, [som2.nbatch, som2.x, som2.y])

        return leak_train_, leak_act_


    
    def make_sparse_leak_act2(self, lang='eng'):

        if self.naming_only:
            raise Lexception("Leaking works only with full connections")

        if lang == 'eng':
            som = self.eng
            act_ = self.prop_act_[self.SEM2ENG]
            wx_ = self.prop_wx_[self.SEM2ENG]
            wy_ = self.prop_wy_[self.SEM2ENG]
        elif lang == 'spa':
            som = self.spa
            act_ = self.prop_act_[self.SEM2SPA]
            wx_ = self.prop_wx_[self.SEM2SPA]
            wy_ = self.prop_wy_[self.SEM2SPA]
            

        leak_winner_indices_ = tf.reshape(tf.stack([som.nbatch_list_, wx_,wy_], axis=-1), [-1,1,3])
        leak_indices_act_ = tf.add(leak_winner_indices_, som.circlei_)
        leak_indices_W_ = leak_indices_act_[:,:,1:]

        raw_sparse_act_ = tf.multiply(tf.gather_nd(act_, leak_indices_act_), 1.0-som.sparse_mask_)
        max_ = tf.gather_nd(act_, leak_winner_indices_)

        sparse_classic_leak_act_ = tf.multiply(raw_sparse_act_ / max_,   #### applying classic norm first, 7/2/18 -uli
                                               1.0-som.sparse_mask_)     #### for now, just divide by the max... see if it fixed leak nans 7/8/28 -U 

        
        sparse_gauss_leak_act_ = tf.multiply(sparse_classic_leak_act_, som.sparse_theta_)   #### applying gauss act now, 6/30/18 -uli

        return leak_indices_act_, leak_indices_W_, sparse_gauss_leak_act_        


    
    def make_leak_train_op2(self, lang='eng'):

        if self.naming_only:
            raise Lexception("Leaking works only with full connections")

        if lang=='eng':
            som2 = self.eng
            som1 = self.spa
            assoc_i = self.SPA2ENG
        else:
            som2 = self.spa
            som1 = self.eng
            assoc_i = self.ENG2SPA
                    
        
        act1_ = som1.sparse_act_
        
        leak_indices_act_, leak_indices_W_, sparse_leak_act_ =  self.make_sparse_leak_act(lang)
        
        deltas_ = self.leak_alpha2_ * tf.multiply(tf.expand_dims(act1_, -1),
                                        tf.expand_dims(sparse_leak_act_, -2))        

        indices12_ = tf.cast(tf.add(tf.constant([[[[assoc_i,0,0,0,0]]]], dtype=self.dtype),
                                    tf.add(tf.pad(tf.expand_dims(tf.cast(som1.indices_W_, self.dtype),
                                                                 -2),
                                                  [[0,0],[0,0],[0,0], [1,2]]),
                                           tf.pad(tf.expand_dims(tf.cast(leak_indices_W_, self.dtype),
                                                                 -3),
                                                  [[0,0],[0,0],[0,0], [3,0]]))),
                            self.itype)

        leak_train_ = tf.scatter_nd_add(self.assoc_, indices12_, deltas_, use_locking=False)   

        leak_act_ = tf.scatter_nd(leak_indices_act_, sparse_leak_act_, [som2.nbatch, som2.x, som2.y])
        
        return leak_train_, leak_act_



    
############################ MORE CONVENIENT MODEL CLASS ###########################

class Lex(lex_):

    logfile = sys.stdout
    
    def __init__(self, sem_data, eng_data, spa_data, x=None, y=None,
                 naming_only=False,
                 sem=None, eng=None, spa=None, assoc=None,
                 act_fun='gauss', name="DISCERN",
                 age=0.0):

        if sem is not None:
            x,y = sem.shape[:2]
            
        lex_.__init__(self, sem_data, eng_data, spa_data, x, y, naming_only=naming_only,
                      act_fun=act_fun, name=name)

        self.start_age = float(age)

        self.log("INFO: start age is %.2f"%self.start_age)
        self.age = None
        
        self.npatterns = 0
        self.neng_patterns = 0
        self.nspa_patterns = 0
        
        self.orig_sem = sem
        self.orig_eng = eng
        self.orig_spa = spa
        self.orig_assoc = assoc
        
        if assoc is not None:
            if self.naming_only:
                if assoc.shape[0] == 6:
                    self.orig_assoc = assoc[[0,2]]
                else:
                    self.orig_assoc = assoc
            elif assoc.shape[0] == 6:
                self.orig_assoc = assoc
            else:
                raise Lexception("Need full assoc array")
            
        self.orig_lesioned = False     
        self.treating = False
        self.treatment_words = None

        
        self.semantic_lesion_type = "none"
        self.phonetic_lesion_type = "none"
        self.assoc_lesion_type = "none"
        
        self.semantic_lesion_strength = None
        self.phonetic_lesion_strength = None
        self.eng_lesion_strength = None
        self.spa_lesion_strength = None
        self.assoc_lesion_strength = None
               
        
    @classmethod    
    def log(cls, msg=".", nl=True, flush=True, echo=True):
        
        echo = echo and not cls.logfile is sys.stdout

        if cls.logfile is not None:
            cls.logfile.write(str(msg))
            if nl:
                cls.logfile.write("\n")
            if flush:
                cls.logfile.flush()
                         
        if echo or cls.logfile is None:
            sys.stdout.write(str(msg))           
            if nl:
                sys.stdout.write("\n")
            if flush:
                sys.stdout.flush()

        
                           
    @classmethod
    def load_npz(cls, fname, P, T, W, naming_only=False):

        cls.log("INFO:   LOADING %s"%fname)
        npz = np.load(fname)
        params = {key:npz[key] for key in npz.keys()}    
        x, y, _ = params['sem'].shape

        print "SHAPE", x,y

        print "naming only: %r"%naming_only

        print "INFO: npz keys %r"%npz.keys()

        if "age" in npz:
            print "INFO: start should be %r"%npz["age"]
                
        try:
            if 'name' in npz:
                name = str(npz["name"])
            else:
                name = fname.replace(".npz", "").replace(".npy", "").split("/")[-1]
        except:
            name = "unnamed"

        if "age" in npz:
            age = float(npz["age"])
        else:
            print "no age in npz"
            age = None

        if "pre_seed" in npz:
            P.pre_seed = int(npz["pre_seed"])
            cls.log("INFO:      npz pre_seed: %d"%(P.pre_seed))
        if "sem_lesion_seed" in npz:
            P.sem_lesion_seed = int(npz["sem_lesion_seed"])
            cls.log("INFO:      npz sem_lesion_seed: %d"%(P.sem_lesion_seed))
        if "eng_lesion_seed" in npz:
            P.eng_lesion_seed = int(npz["eng_lesion_seed"])
            cls.log("INFO:      npz eng_lesion_seed: %d"%(P.eng_lesion_seed))
        if "spa_lesion_seed" in npz:
            P.spa_lesion_seed = int(npz["spa_lesion_seed"])
            cls.log("INFO:      npz spa_lesion_seed: %d"%(P.spa_lesion_seed))
        if "post_seed" in npz:
            P.post_seed = int(npz["post_seed"])
            cls.log("INFO:      npz post_seed: %d"%(P.post_seed))
                          
        model = cls(W.sem_data, W.eng_data, W.spa_data, x=x, y=y, naming_only=naming_only, eng=npz["eng"], spa=npz["spa"], sem=npz["sem"], assoc=npz["assoc"], name=name, age=age)
        
        if "lesioned" in npz and bool(npz["lesioned"]):

            cls.log("INFO:   Loaded lexicon is lesioned!")

            model.orig_lesioned = True         

            model.semantic_lesion_type = str(npz["semantic_lesion_type"])
            model.phonetic_lesion_type = str(npz["phonetic_lesion_type"])
        
            model.semantic_lesion_strength = float(npz["semantic_lesion_strength"])
            model.eng_lesion_strength =  float(npz["eng_lesion_strength"])
            model.spa_lesion_strength =  float(npz["spa_lesion_strength"])        

            model.semantic_lesion_mask = npz["semantic_lesion_mask"]
            model.eng_lesion_mask = npz["eng_lesion_mask"]
            model.spa_lesion_mask = npz["spa_lesion_mask"]
            
            
            cls.log("INFO:   semantic: %r, %r"%(model.semantic_lesion_type, model.semantic_lesion_strength))
            cls.log("INFO:   phonetic: %r, %r / %r"%(model.phonetic_lesion_type, model.eng_lesion_strength, model.spa_lesion_strength))
            
        else:
            cls.log("INFO:   Loaded lexicon is NOT lesioned!")

        return model


    
    
    def save_npz(self, file_name, P, T, sess=None):
        
        sess = sess if sess is not None else tf.get_default_session()        

        names = ['sem','eng','spa','assoc']
                    
        params = sess.run([self.sem.W_, self.eng.W_, self.spa.W_,
                           self.assoc_])

        if P.pre_seed is not None:
            names.append("pre_seed")
            params.append(P.pre_seed)
        if self.lesioned and P.sem_lesion_seed is not None:
            names.append("sem_lesion_seed")
            params.append(P.sem_lesion_seed)
        if self.lesioned and P.eng_lesion_seed is not None:
            names.append("eng_lesion_seed")
            params.append(P.eng_lesion_seed)
        if self.lesioned and P.spa_lesion_seed is not None:
            names.append("spa_lesion_seed")
            params.append(P.spa_lesion_seed)
        if self.lesioned and P.post_seed is not None:
            names.append("post_seed")
            params.append(P.post_seed)
        if self.treating and T.treatment_seed is not None:
            names.append("treatment_seed")
            params.append(T.treatment_seed)
            names.append("treatment_words")
            params.append(self.treatment_words)  #just in case?

                    
        np.savez_compressed(file_name,
                            age=self.age, act_fun=self.act_fun, name=self.name,
                            lesioned = self.lesioned,
                            semantic_lesion_type = self.semantic_lesion_type, phonetic_lesion_type = self.phonetic_lesion_type,
                            semantic_lesion_strength=self.semantic_lesion_strength, eng_lesion_strength = self.eng_lesion_strength, spa_lesion_strength = self.spa_lesion_strength,
                            semantic_lesion_mask=self.semantic_lesion_mask, eng_lesion_mask = self.eng_lesion_mask, spa_lesion_mask = self.spa_lesion_mask,
                            **dict(zip(names,params)))    

        

    def init_model(self, P, H, sess=None):

        t0=time.time()
        
        sess = sess if sess is not None else tf.get_default_session()

        # reset age
        self.age = self.start_age
        self.npatterns = self.neng_patterns = self.nspa_patterns = 0

        self.treating = False
        
        # reset weights
        for src,tgt in zip([self.orig_sem,self.orig_eng, self.orig_spa, self.orig_assoc],
                           [self.sem.W_, self.eng.W_, self.spa.W_, self.assoc_]):
            if src is not None:
                sess.run(tf.assign(tgt, src))
            else:
                sess.run(tf.variables_initializer([tgt]))

        # reset lesion
        
        if not self.orig_lesioned:
            self.lesioned = False
            self.semantic_lesion_type =  P.semantic_lesion_type
            self.phonetic_lesion_type = P.phonetic_lesion_type
        
            self.semantic_lesion_strength = 0.0
            self.eng_lesion_strength = 0.0
            self.spa_lesion_strength = 0.0

            self.semantic_lesion_mask = np.array([[0.0]], dtype=np.float32)        
            self.eng_lesion_mask = np.array([[0.0]], dtype=np.float32)
            self.spa_lesion_mask = np.array([[0.0]], dtype=np.float32)
                                 
        else:
            self.lesioned = True
            self.semantic_lesion_type =  P.semantic_lesion_type
            self.phonetic_lesion_type = P.phonetic_lesion_type            

        
        
    ######################## TESTING ################################
            
    def present_and_prop_all(self, P, test_trans=False, sess=None):

        sess = sess if sess is not None else tf.get_default_session()  

        t = time.time()
        
        columns = ['semx', 'semy', 'engx', 'engy', 'spax', 'spay', 
                   'enx', 'eny', 'snx', 'sny']

            
        ops = [self.sem.wx_, self.sem.wy_,
               self.eng.wx_, self.eng.wy_,
               self.spa.wx_, self.spa.wy_,
               self.prop_wx_[self.SEM2ENG],
               self.prop_wy_[self.SEM2ENG],
               self.prop_wx_[self.SEM2SPA],
               self.prop_wy_[self.SEM2SPA]] 

        if test_trans:

            self.log("INFO:     prop_all: testing translation")
            columns += ['s2ex', 's2ey', 'e2sx', 'e2sy']
            
            ops += [self.prop_wx_[self.SPA2ENG],
                    self.prop_wy_[self.SPA2ENG],
                    self.prop_wx_[self.ENG2SPA],
                    self.prop_wy_[self.ENG2SPA]]
            
        N = self.ninp
        subset = np.arange(N)    
            
        data = np.zeros((len(columns),N), dtype=np.int32)

        sigmaf = P.sigmaf()
        
        for p in range(0, N, P.nbatch):

            batch = subset[p:min(p+P.nbatch,len(subset))]
            
            feed = {self.i_: batch, 
                    self.sigma_: sigmaf(self.age), 
                    self.R_:max(P.minR, 3.0*sigmaf(self.age)),
                    self.sem.mask_: self.semantic_lesion_mask,
                    self.eng.mask_: self.eng_lesion_mask,
                    self.spa.mask_: self.spa_lesion_mask
                    }

            data[:,batch] = np.array(sess.run(ops, feed))

        self.log("present and prop took %.3f s"%(time.time()-t))
            
        return pd.DataFrame(index=self.sem_data.index,
                            columns = columns,
                            data=data.T)

        
    def score_naming(self, res, W, subset=None, sess=None, return_results=False, test_trans=False):

        sess = sess if sess is not None else tf.get_default_session()

        t = time.time()
        
        df = res

        scores = pd.DataFrame(index=df.index, data={
        'sem_mult': df.groupby(['semx', 'semy']).size().ix[zip(df.semx, df.semy)].tolist(),
        'eng_mult': df.groupby(['engx', 'engy']).size().ix[zip(df.engx, df.engy)].tolist(),
        'spa_mult': df.groupby(['spax', 'spay']).size().ix[zip(df.spax, df.spay)].tolist(),

        'en_corr': ((df.enx == df.engx) & (df.eny == df.engy)).astype(int),
        'sn_corr': ((df.snx == df.spax) & (df.sny == df.spay)).astype(int)})

        if test_trans:
            self.log("INFO:     naming: testing translation")
            scores['e2s_corr'] = ((df.e2sx == df.spax) & (df.e2sy == df.spay)).astype(int)  
            scores['s2e_corr'] = ((df.s2ex == df.engx) & (df.s2ey == df.engy)).astype(int)   

        res["en_corr"] = scores.loc[res.index, 'en_corr']
        res["sn_corr"] = scores.loc[res.index, 'sn_corr']
        res["eng_mult"] = scores.loc[res.index, 'eng_mult']
        res["spa_mult"] = scores.loc[res.index, 'spa_mult']

        if test_trans:
            res["et_corr"] = scores.loc[res.index, 'e2s_corr']
            res["st_corr"] = scores.loc[res.index, 's2e_corr']

        
        self.log("naming took %.3f s"%(time.time()-t))

        
        if not return_results:

            eng_scr = (scores.loc[W.common_words, 'en_corr'] / scores.loc[W.common_words, 'eng_mult']).mean()
            spa_scr = (scores.loc[W.common_words, 'sn_corr'] / scores.loc[W.common_words, 'spa_mult']).mean()
            eng_bnt = (scores.loc[W.rare_words, 'en_corr'] / scores.loc[W.rare_words, 'eng_mult']).mean()
            spa_bnt = (scores.loc[W.rare_words, 'sn_corr'] / scores.loc[W.rare_words, 'spa_mult']).mean()

            if subset is None:
                eng_subset, spa_subset = -1, -1
            else:            
                eng_subset = (scores.loc[subset, 'en_corr'] / scores.loc[subset, 'eng_mult']).mean()
                spa_subset = (scores.loc[subset, 'sn_corr'] / scores.loc[subset, 'spa_mult']).mean()
            
            return eng_scr, spa_scr, eng_bnt, spa_bnt, eng_subset, spa_subset
        
        else:
            return res


    def score_naming_alt(self, res, P, W, sess=None, subset=None, return_results=False, test_trans=False):
        
        sess = sess if sess is not None else tf.get_default_session()

        t = time.time()

        N = self.ninp
                    
        sigmaf = P.sigmaf()

        feed = {self.i_: np.arange(0, N, dtype=np.int32),
                self.R_: max(1.0, 3.0*sigmaf(self.age)),
                self.sem.mask_: self.semantic_lesion_mask,
                self.eng.mask_: self.eng_lesion_mask,
                self.spa.mask_: self.spa_lesion_mask
                }

        

        eng_labels = sess.run(self.eng.labels_, feed)
        spa_labels = sess.run(self.spa.labels_, feed)
        
        res["en_label"] = eng_labels[res.enx.values,res.eny.values]
        res["sn_label"] = spa_labels[res.snx.values,res.sny.values]
        
        res["en_corr_alt"] = (res["en_label"]==np.arange(0, N, dtype=np.int32))
        res["sn_corr_alt"] = (res["sn_label"]==np.arange(0, N, dtype=np.int32))

        if test_trans:
            self.log("INFO:     alt_name: testing translation")
            res["st_label"] = spa_labels[res.e2sx.values,res.e2sy.values]
            res["et_label"] = eng_labels[res.s2ex.values,res.s2ey.values]
            res["st_corr_alt"] = (res["st_label"]==np.arange(0, N, dtype=np.int32))
            res["et_corr_alt"] = (res["et_label"]==np.arange(0, N, dtype=np.int32))
        
        res["common"] = res.index.isin(W.common_words)
        
        eng_scr = res.en_corr_alt[W.common_words].mean()
        eng_bnt = res.en_corr_alt[W.rare_words].mean()
        spa_scr = res.sn_corr_alt[W.common_words].mean()
        spa_bnt = res.sn_corr_alt[W.rare_words].mean()

        if subset is None:
            eng_subset, spa_subset = -1, -1
        else:            
            eng_subset = res.en_corr_alt[subset].mean()
            spa_subset = res.sn_corr_alt[subset].mean()
      
        self.log("alt naming took %.3f s"%(time.time()-t))

        if not return_results:
            return eng_scr, spa_scr, eng_bnt, spa_bnt, eng_subset, spa_subset
        else:
            return res

        
    
    def score_papt(self, W, thresh=0.0, nbatch=32, sess=None):

        t =  time.time()
        
        sess = sess or tf.get_default_session()
        papt3 = W.papt3.reset_index(drop=True)

        
        N = len(papt3)
        f1 = np.zeros(N, dtype=np.float32)
        f2 = np.zeros(N, dtype=np.float32)
        fc = np.zeros(N, dtype=np.float32)

        for p in range(0, N, nbatch):

            batch = np.arange(p, min(p+nbatch,N))

            f1[batch] = sess.run(self.sem.papt_guess_, {self.i_: papt3.loc[batch,"nw1"].values,
                                                        self.sem.papt_cues_: papt3.loc[batch,"nf"].values,
                                                        self.sem.mask_: self.semantic_lesion_mask})
            f2[batch] = sess.run(self.sem.papt_guess_, {self.i_: papt3.loc[batch,"nw2"].values,
                                                        self.sem.papt_cues_: papt3.loc[batch,"nf"].values,
                                                        self.sem.mask_: self.semantic_lesion_mask})
            fc[batch] = sess.run(self.sem.papt_guess_, {self.i_: papt3.loc[batch,"ncue"].values,
                                                        self.sem.papt_cues_: papt3.loc[batch,"nf"].values,
                                                        self.sem.mask_: self.semantic_lesion_mask})
        papt3["f1"] = f1
        papt3["f2"] = f2
        papt3["fcue"] = fc

        papt3["fscore"] = (papt3["f2"]-papt3["fcue"]).abs()-(papt3["f1"]-papt3["fcue"]).abs()
        papt3["fscore"] = papt3["fscore"].abs()

        res = papt3.sort_values("fscore", ascending=False).groupby(["nw1", "nw2", "ncue"]).first()


        res["correct"] = 0.5

        # correct answer is always B
        res.loc[(res.f1-res.fcue).abs()>(res.f2-res.fcue).abs(), "correct"] = 0.0
        res.loc[(res.f2-res.fcue).abs()>(res.f1-res.fcue).abs(), "correct"] = 1.0

        ### APPLY THRESHOLD
        res.loc[res.fscore<thresh, "correct"] = 0.5

        self.log("INFO:     papt took %.3f s"%(time.time()-t))
        
        return res.correct.mean() 



    def get_nan_counts(self, assoc_only=True, sess=None):
            
        sess = sess if sess is not None else tf.get_default_session()

        self.log("INFO:     counting assoc NaNs")
        
        if assoc_only:
            return sess.run(tf.cast(tf.reduce_sum(tf.cast(tf.is_nan(self.assoc_), tf.float32)), tf.int32))
        else:
            return  sess.run([tf.cast(tf.reduce_sum(tf.cast(tf.is_nan(ww), tf.float32)), tf.int32) for ww in [self.assoc_[SEM2ENG],
                                                                                                              self.assoc_[SEM2SPA],
                                                                                                              self.assoc_[ENG2SPA],
                                                                                                              self.assoc_[SPA2ENG],
                                                                                                              self.sem.W_,
                                                                                                              self.eng.W_,
                                                                                                              self.spa.W_]])


    
    def test_all(self, P, W, H, T, sess=None, v=True, test_trans=False, **args):
        
        sess = sess if sess is not None else tf.get_default_session()

        
        context_index = args.keys() + ['hid','pi','ti','age','lesioned','treating','treat_lang','npat', 'eng_pat','spa_pat',
                                       'pre_seed','sem_lesion_seed','eng_lesion_seed','spa_lesion_seed','post_seed','treatment_seed',
                                       'lesion_type', 'sem_lesion_strength', 'eng_lesion_strength', 'spa_lesion_strength',
                                       'nan_count']
                 
        context = args.values() + [H.hid, P.pi, T.ti, self.age, self.lesioned, self.treating, H.treatment_language,
                                   self.npatterns, self.neng_patterns, self.nspa_patterns,
                                   P.pre_seed, P.sem_lesion_seed, P.eng_lesion_seed, P.spa_lesion_seed, P.post_seed, T.treatment_seed,
                                   self.semantic_lesion_type, self.semantic_lesion_strength, self.eng_lesion_strength, self.spa_lesion_strength,
                                   self.get_nan_counts(sess=sess)]
               
        results_index = ['eng_scr', 'spa_scr', 'eng_bnt', 'spa_bnt', 'eng_subset', 'spa_subset',
                         'alt_eng_scr', 'alt_spa_scr', 'alt_eng_bnt', 'alt_spa_bnt', 'alt_eng_subset', 'alt_spa_subset',
                         "PAPT", "PAPT_THRESH"]
                             
        index = context_index + results_index


        

        t = time.time()


        test_trans = test_trans and not self.naming_only

        
        res = self.present_and_prop_all(P, sess=sess, test_trans=test_trans)

        if self.treating:
            subset = self.treatment_words
        else:
            subset = None


        
            
        eng_scr, spa_scr, eng_bnt, spa_bnt, eng_subset, spa_subset = self.score_naming(res, W, subset=subset, sess=sess, test_trans=test_trans)
        alt_eng_scr, alt_spa_scr, alt_eng_bnt, alt_spa_bnt, alt_eng_subset, alt_spa_subset = self.score_naming_alt(res, P, W,
                                                                                                                   subset=subset, sess=sess,
                                                                                                                   test_trans=test_trans)
        
        papt_score = -1
        papt_score_thresh = -1
        
        if W.papt3 is not None:        
            papt_score = self.score_papt(W, thresh=0.0, nbatch=32, sess=sess)

        if W.papt3 is not None and P.papt_thresh > 0.0:        
            papt_score_thresh = self.score_papt(W, P.papt_thresh, nbatch=32, sess=sess)

            
        results = [eng_scr, spa_scr, eng_bnt, spa_bnt, eng_subset, spa_subset,
                   alt_eng_scr, alt_spa_scr, alt_eng_bnt, alt_spa_bnt, alt_eng_subset, alt_spa_subset,
                   papt_score, papt_score_thresh]

        if v:
            self.log(''.join(["%-15s "]*14)%tuple(index[-14:]))
            self.log(''.join(["%-15.2f "]*14)%tuple(results))

        self.log("test_all took %.3f s"%(time.time()-t))

        for name, val in zip(context_index, context):
            res[name] = val
        res["word"] = res.index
        if self.treating:
            res["treatment_word"] = res.word.isin(self.treatment_words)
       
        return pd.Series(index = index, data = context + results), res





    ######################### TRAINING  ###############################

   
    def noise_assoc(self, P, fraction=1.0, sess=None):
        
        sess = sess or tf.get_default_session()


        if P.noise in ["gaussian", "normal"]:
            ### variance is the sum of the two variances
            var = P.noise_std * P.noise_std * fraction        
            sess.run(self.noise_assoc_normal_, {self.noise_std_: sqrt(var)})
        elif P.noise in ["classic", "uniform"]:
            std =  P.noise_std*fraction 
            sess.run(self.noise_assoc_uniform_, {self.noise_std_: std})

        
    def norm_assoc(self, P, sess=None):
        
        sess = sess or tf.get_default_session()

        self.log("INFO: running norm_assoc_classic_...")
        sess.run(self.norm_assoc_classic_)

        
        if P.norm != "classic":
             self.log("WARNING:       running norm_assoc_classic_")
             self.log("WARNING:       not doing the others anymore...")
        

             
    def train(self, target_age, P, H, W, sess=None, alpha_factor=1.0):
        
        sess = sess or tf.get_default_session()

        t = time.time()
        age0 = self.age
        
        N =  W.nwords
        word_freq = W.word_freq()
        
        inputs = np.random.choice(N, size=int(N*P.exposure), p=word_freq.values, replace=True)         
        i=0

        dt = float(P.nbatch)/N/P.exposure

        sigmaf = P.sigmaf()
        map_alphaf = P.map_alphaf()
        assoc_alphaf = P.assoc_alphaf()

        eng_expf = H.eng_expf(self.lesioned)

        
        while(self.age<target_age):
            
            batch = inputs[i:min(i+P.nbatch,len(inputs))]

            self.log("INFO: current age = %r"%self.age)
            
            feed =  {self.i_: batch,
                     self.sigma_: sigmaf(self.age),
                     self.map_alpha_: map_alphaf(self.age) * alpha_factor,
                     self.alpha_: assoc_alphaf(self.age) * alpha_factor,
                     self.R_: max(min(P.hood_std_devs*sigmaf(self.age), P.maxR), P.minR),
                     self.sem.mask_: self.semantic_lesion_mask,
                     self.eng.mask_: self.eng_lesion_mask,
                     self.spa.mask_: self.spa_lesion_mask,
                    }
                        
            train_eng = eng_expf(self.age) > np.random.random()
            train_spa = (1.0-eng_expf(self.age)) > np.random.random()

            self.npatterns += len(batch)
            
            if train_eng and train_spa:
                op = self.train_all
                self.neng_patterns += len(batch)
                self.nspa_patterns += len(batch)
                        
            elif train_eng:
                if self.naming_only or P.train_across_method == "and":
                    op = self.train_english
                elif P.train_across_method == "p":
                    op = self.train_english_p
                elif P.train_across_method == "q":
                    op = self.train_english_q
                elif P.train_across_method == "or":
                    op = self.train_english_or
                    
                self.neng_patterns += len(batch)
                        
            elif train_spa:
                if self.naming_only or P.train_across_method == "and":
                    op = self.train_spanish 
                elif P.train_across_method == "p":
                    op = self.train_spanish_p
                elif P.train_across_method == "q":
                    op = self.train_spanish_q
                elif P.train_across_method == "or":
                    op = self.train_spanish_or
                self.nspa_patterns += len(batch)
                
            else:
                op = self.sem.train_
                        
            _  = sess.run(op, feed)
                        
            i += P.nbatch
            self.age += dt

        self.log("training %.2f years took %.3f s"%(self.age-age0, time.time()-t))

        return self.age
        

    

    ######################## LESIONS ################################        
   
    def apply_lesion(self, P, H, W, T, sess=None):
            
        sess = sess if sess is not None else tf.get_default_session()

        if self.lesioned:
            self.log("INFO:   (apply_lesion:) already lesioned!")
            self.log("INFO:   (NOT lesioning again, duh)")
            raise Lexception("INFO:   (apply_lesion:) already lesioned!")

                        
        self.lesioned = True
        self.semantic_lesion_type = P.semantic_lesion_type
        self.semantic_lesion_strength = P.semantic_lesion_strength
        self.phonetic_lesion_type = P.phonetic_lesion_type
        self.eng_lesion_strength = P.eng_lesion_strength
        self.spa_lesion_strength = P.spa_lesion_strength


        ########## SEM LESION
        

        ### random seed for semantic lesion
        tf.set_random_seed(P.sem_lesion_seed)
        np.random.seed(P.sem_lesion_seed)              

        
        if self.semantic_lesion_type == "gauss_mask":
            self.semantic_lesion_mask = lesion_mask_gauss([self.sem.x, self.sem.y], self.semantic_lesion_strength)
        elif self.semantic_lesion_type == "round_mask":
            self.semantic_lesion_mask = lesion_mask_round([self.sem.x, self.sem.y], self.semantic_lesion_strength)
        elif self.semantic_lesion_type == "none":
            self.semantic_lesion_mask = np.zeros([self.x, self.y], dtype=np.float32) 
        else:
            raise Lexception("unknown lesion type: " + self.semantic_lesion_type)  

        ########## ENG LESION
            

        ### random seed for eng lesion
        tf.set_random_seed(P.eng_lesion_seed)
        np.random.seed(P.eng_lesion_seed)   
                                       
        if self.phonetic_lesion_type == "gauss_mask":
            self.eng_lesion_mask = lesion_mask_gauss([self.x, self.y], self.eng_lesion_strength)
        elif self.phonetic_lesion_type == "round_mask":
            self.eng_lesion_mask = lesion_mask_round([self.x, self.y], self.eng_lesion_strength)
        elif self.phonetic_lesion_type == "none":
            self.eng_lesion_mask = np.zeros([self.x, self.y], dtype=np.float32) 
        else:
            raise Lexception("unknown lesion type: " + self.phonetic_lesion_type)  

        ########## SPA LESION
        
        ### random seed forspa lesion
        tf.set_random_seed(P.spa_lesion_seed)
        np.random.seed(P.spa_lesion_seed)   
                                       
        if self.phonetic_lesion_type == "gauss_mask":
            self.spa_lesion_mask = lesion_mask_gauss([self.x, self.y], self.spa_lesion_strength)
        elif self.phonetic_lesion_type == "round_mask":
            self.spa_lesion_mask = lesion_mask_round([self.x, self.y], self.spa_lesion_strength)
        elif self.phonetic_lesion_type == "none":
            self.spa_lesion_mask = np.zeros([self.x, self.y], dtype=np.float32)        
        else:
            raise Lexception("unknown lesion type: " + self.phonetic_lesion_type)        
   
        ## SEM2ENG = 0
        ## ENG2SEM = 1   
        ## SEM2SPA = 2
        ## SPA2SEM = 3 
        ## ENG2SPA = 4       
        ## SPA2ENG = 5

        if P.mask_assoc and not self.naming_only:
            self.log("INFO:     Masking assoc connections...")
            
            src_mask_ = 1.0 - tf.reshape(tf.stack([self.semantic_lesion_mask,
                                                   self.eng_lesion_mask,
                                                   self.semantic_lesion_mask,
                                                   self.spa_lesion_mask,
                                                   self.eng_lesion_mask,
                                                   self.spa_lesion_mask],
                                                   axis=0),
                                         [6, self.x, self.y, 1, 1])

            tgt_mask_ = 1.0 - tf.reshape(tf.stack([self.eng_lesion_mask,
                                                   self.semantic_lesion_mask,
                                                   self.spa_lesion_mask,
                                                   self.semantic_lesion_mask,
                                                   self.spa_lesion_mask,
                                                   self.eng_lesion_mask],
                                                   axis=0),
                                         [6, 1, 1, self.x, self.y])            
            
            
            tf.assign(self.assoc_, self.assoc_ * tf.cast(src_mask_, tf.float32) * tf.cast(tgt_mask_, tf.float32) )
            
        elif P.mask_assoc:
            self.log("INFO:     naming_only, no need to mask assoc connections...")
                   
        else:
            self.log("INFO:     NOT masking assoc connections...")
            
    

    #####################  TREATMENT   #########################


    
    def start_treatment(self, P, H, W, T, sess=None):
        
        sess = sess if sess is not None else tf.get_default_session()
        
        if self.treating:
            print "ALREADY TREATING!!!"
            raise Lexception("ALREADY TREATING!!!")


        self.log("INFO:    Starting Treatment!!!!")
        
        tmp = self.present_and_prop_all(P, sess=sess)        
        res = self.score_naming_alt(tmp, P, W, sess=sess, return_results=True)

        
        if T.select_words == "random":
            self.treatment_words = np.random.choice(W.common_words,
                                                    size=T.ntreatment_words,
                                                    replace=False)
        else:
            tmp = self.present_and_prop_all(P, sess)        
            res = self.score_naming_alt(tmp, P, W, sess=sess, return_results=True)

            if T.select_words == "qpu":

                base_eng = H.baseline_eng
                base_spa = H.baseline_spa

                opt_eng = res[res.index.isin(W.common_words) & (res.en_corr_alt) & (~res.sn_corr_alt)].index.tolist()
                opt_spa = res[res.index.isin(W.common_words) & (~res.en_corr_alt) & (res.sn_corr_alt)].index.tolist()
                opt_neither = res[res.index.isin(W.common_words) & (~res.en_corr_alt) & (~res.sn_corr_alt)].index.tolist()

                
                if len(opt_eng)>0 and base_eng>0.0:
                    eng = np.random.choice(opt_eng, size=min(len(opt_eng), int(T.ntreatment_words*base_eng)), replace=False)
                else:
                    eng = np.array([])
                    
                if len(opt_spa)>0 and base_spa>0.0:                    
                    spa = np.random.choice(opt_spa, size=min(len(opt_spa), int(T.ntreatment_words*base_spa)), replace=False)
                else:
                    spa = np.array([])
                                                           
                neither = np.random.choice(opt_neither,
                                           size=min(len(opt_neither),int(T.ntreatment_words-len(eng)-len(spa))),
                                           replace=False)

                self.treatment_words = np.concatenate([neither,eng,spa])

                
            elif T.select_words == "neither":
                opt_neither = res[res.index.isin(W.common_words) & (~res.en_corr_alt) & (~res.sn_corr_alt)].index.tolist()

                self.treatment_words = np.random.choice(opt_neither,
                                                        size=min(len(opt_neither),int(T.ntreatment_words)),
                                                        replace=False)
                
            else:
                raise Lexception("What kind of language is %s??"%(H.treatment_language))
                
        self.log("INFO: TREATMENT LANGUAGE IS: %r"%H.treatment_language)
            
        

        self.log("INFO: TREATMENT WORDS")
        self.log("INFO: %r"%self.treatment_words)

        self.treating = True
            


    def treat_one_session(self, P, H, W, T, sess=None):

        sess = sess if sess is not None else tf.get_default_session()
        
        if not self.treating:
            print "should be treating, no?"
            raise Lexception("should be treating, no?")
        

        sigmaf = P.sigmaf()
        map_alphaf = P.map_alphaf()
        assoc_alphaf = P.assoc_alphaf()

        eng_expf = H.eng_expf(True)

        batch = W.word2int(self.treatment_words)       
        
        # train treatment_words in semantic map
        
        feed =  {self.i_: batch,
                 self.sigma_: sigmaf(self.age),
                 self.map_alpha_: map_alphaf(self.age) * T.treatment_sem_factor,
                 self.R_: max(min(T.treatment_hood_size * sigmaf(self.age), P.maxR), P.minR),
                 self.sem.mask_: self.semantic_lesion_mask}

        _  = sess.run(self.sem.train_, feed)
        
        label_feed = {self.i_: np.arange(0, self.ninp, dtype=np.int32),
                      self.R_: max(1.0, 3.0*sigmaf(self.age)),
                      self.sem.mask_: self.semantic_lesion_mask,
                      self.eng.mask_: self.eng_lesion_mask,
                      self.spa.mask_: self.spa_lesion_mask
                     }

        spa_labels = sess.run(self.spa.labels_, label_feed)
        eng_labels = sess.run(self.eng.labels_, label_feed) 


        treat_feed =  {self.i_: batch,
                       self.sigma_: sigmaf(self.age),
                       self.map_alpha_: map_alphaf(self.age) * T.treatment_alpha_factor,
                       self.alpha_: assoc_alphaf(self.age) * T.treatment_alpha_factor,
                       self.leak_alpha_: assoc_alphaf(self.age) * T.treatment_leak_factor,
                       self.leak_alpha2_: assoc_alphaf(self.age) * T.treatment_leak_factor2,
                       self.R_: max(min(T.treatment_hood_size*sigmaf(self.age), P.maxR), P.minR),
                       self.sem.mask_: self.semantic_lesion_mask,
                       self.eng.mask_: self.eng_lesion_mask,
                       self.spa.mask_: self.spa_lesion_mask,
                       }

        if T.noleak and H.treatment_language == 'eng':   ### training english, NOT leaking
            
            sess.run(self.train_english, treat_feed)
                     
        elif T.noleak and H.treatment_language == 'spa':   ### training spanish, NOT leaking
                        
            sess.run(self.train_spanish, treat_feed)
                             
        elif H.treatment_language == 'eng':   ### training english AND leaking
            
            _, twx, twy, nwx, nwy  = sess.run([self.train_english,
                                               self.prop_wx_[self.ENG2SPA],
                                               self.prop_wy_[self.ENG2SPA],
                                               self.prop_wx_[self.SEM2ENG],
                                               self.prop_wy_[self.SEM2ENG]], treat_feed)
            
           
            
            if T.leak_filter == 'trans':  ### note: trying to use only correctly LEAKED, not named
                self.log("INFO:    leaking correctly translated...")
                leak_batch = [n for i,n in enumerate(batch) if (spa_labels[twx[i],twy[i]] == n)]
            elif T.leak_filter == 'named':
                self.log("INFO:    leaking correctly named...")
                leak_batch = [n for i,n in enumerate(batch) if (eng_labels[nwx[i],nwy[i]] == n)]   
            else:
                leak_batch = batch
                self.log("INFO:    leaking ALL")

            self.log("INFO:    leaking %d of %d..."%(len(leak_batch), len(batch)))

            if len(leak_batch) > 0:

                treat_feed[self.i_] = leak_batch
                self.log("INFO:      leaked batch: %r"%(treat_feed[self.i_]))


                if T.leak_use_fake_act:

                    treat_feed[self.map_alpha_] = 0.0                      
                    treat_feed[self.alpha_] = assoc_alphaf(self.age) * T.treatment_leak_factor

                    self.log("INFO:    USING FAKE LEAKED ACT for SPA, assoc alpha = %f"%(treat_feed[self.alpha_]))

                    _ = sess.run([self.train_spanish], treat_feed)

                    if T.leak_train_map_factor>0.0:
                        self.log("WARNING:    MAP TRAINING NOT IMPLEMENTED")
                            
                else:
                    self.log("INFO:    USING REAL LEAKED ACT")    



                    ### then train non-treated map using leak winners... 
                    if T.leak_train_map_factor > 0.0:
                        _, cross_winners = sess.run([self.leak_train_spa_, #  self.leak_train_spa2_,   #### turning leak2 OFF for now
                                                    self.prop_winner_vectors_[self.ENG2SPA]], treat_feed)                        
                        _ = sess.run(self.spa.train_, {self.spa.alpha_: map_alphaf(self.age) * T.leak_train_map_factor,
                                                       self.spa.input_: cross_winners,
                                                       self.spa.sigma_:sigmaf(self.age),
                                                       self.spa.R_: max(min(T.treatment_hood_size*sigmaf(self.age), P.maxR), P.minR),
                                                       self.spa.mask_: self.spa_lesion_mask,})
                    else:
                        _ = sess.run([self.leak_train_spa_], #  self.leak_train_spa2_,   #### turning leak2 OFF for now
                                     treat_feed)                        

        elif H.treatment_language == 'spa':   ### training spanish AND leaking
           
            _, twx, twy, nwx, nwy = sess.run([self.train_spanish,
                                              self.prop_wx_[self.SPA2ENG],
                                              self.prop_wy_[self.SPA2ENG],
                                              self.prop_wx_[self.SEM2SPA],
                                              self.prop_wy_[self.SEM2SPA]], treat_feed)

            if T.leak_filter == 'trans':  ### note: trying to use only correctly LEAKED, not named
                self.log("INFO:    leaking correctly translated...")
                leak_batch = [n for i,n in enumerate(batch) if (eng_labels[twx[i],twy[i]] == n)]
            elif T.leak_filter == 'named':
                self.log("INFO:    leaking correctly named...")
                leak_batch = [n for i,n in enumerate(batch) if (spa_labels[nwx[i],nwy[i]] == n)]                
            else:
                leak_batch = batch
                self.log("INFO:    leaking ALL")
                
            self.log("INFO:    leaking %d of %d..."%(len(leak_batch), len(batch)))

            if len(leak_batch) > 0:

                treat_feed[self.i_] = leak_batch
                self.log("INFO:      leaked batch: %r"%(treat_feed[self.i_]))


                if T.leak_use_fake_act:

                    treat_feed[self.map_alpha_] = 0.0
                    treat_feed[self.alpha_] = assoc_alphaf(self.age) * T.treatment_leak_factor
                    self.log("INFO:    USING FAKE LEAKED ACT for ENG, assoc alpha = %f"%(treat_feed[self.alpha_]))

                    _ = sess.run([self.train_english], treat_feed)

                    if T.leak_train_map_factor>0.0:
                        self.log("WARNING:    MAP TRAINING NOT IMPLEMENTED")                            

                else:
                    self.log("INFO:    USING REAL LEAKED ACT")    



                    ### then train non-treated map using leak winner... 
                    if T.leak_train_map_factor > 0.0:
                        _, cross_winners = sess.run([self.leak_train_eng_, #self.leak_train_eng2_,   #### turning leak2 OFF for now
                                                     self.prop_winner_vectors_[self.SPA2ENG]], treat_feed)
                        _ = sess.run(self.eng.train_, {self.eng.alpha_: map_alphaf(self.age) * T.leak_train_map_factor,
                                                       self.eng.input_: cross_winners,
                                                       self.eng.sigma_:sigmaf(self.age),
                                                       self.eng.R_: max(min(T.treatment_hood_size*sigmaf(self.age), P.maxR), P.minR),
                                                       self.eng.mask_: self.eng_lesion_mask,})
                    else:
                        _ = sess.run([self.leak_train_eng_], #self.leak_train_eng2_,   #### turning leak2 OFF for now
                                     treat_feed)

                
        self.age += 1.0/(52.0*T.sessions_per_week)
