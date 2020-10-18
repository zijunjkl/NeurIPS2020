# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:40:14 2019

@author: Zijun Cui
"""
import tensorflow as tf
import pdb
"""  Importing and running isolated TF graph """	  
    
class ImportRightAU():	        
    def __init__(self, model_path):	        
        # Create local graph and use it in the session	        
        self.graph = tf.Graph()	        
        self.sess = tf.Session(graph=self.graph)	        
        with self.graph.as_default():	            
            # Import saved model from location 'loc' into local graph	            
            meta_graph = tf.train.import_meta_graph(model_path + '.meta')	     	                                                                                                    
            meta_graph.restore(self.sess, model_path)	 
            print('Model Restored')	              
            # FROM SAVED COLLECTION:            	            
            self.activation = tf.get_collection('activation') #[p_AUs, pred_AUs, loss_AU_p1, train_AU_p1, loss_AU_p2, train_AU_p2]

    def run(self, image, AUprob, AUconfig):	        
        feed = {'x_image_orig:0': image, 'keep_prob:0': 1, 'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig}
        p_AU, pred_AU, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[2]], feed_dict = feed)
        return p_AU, pred_AU, loss

    def run_p2(self, image, AUprob, AUconfig, PGM_p_AU, posterior, balance_w):	        
        feed = {'x_image_orig:0': image, 'keep_prob:0': 1, 'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig, 'PGM_p_AUconfig:0': PGM_p_AU, 'posterior:0':posterior, 'balance_w:0':balance_w}
        p_AU, pred_AU, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[4]], feed_dict = feed)
        return p_AU, pred_AU, loss
    
    def train(self, image, AUprob, learning_rate, AUconfig):	    
        feed = {'x_image_orig:0': image, 'keep_prob:0': 0.5, 'learning_rate:0': learning_rate, 'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig}
        self.sess.run(self.activation[3], feed_dict = feed)

    def train_p2(self, image, AUprob, learning_rate, AUconfig, PGM_p_AU, posterior, balance_w):	    
        feed = {'x_image_orig:0': image, 'keep_prob:0': 1, 'learning_rate:0': learning_rate, 'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig, 'PGM_p_AUconfig:0': PGM_p_AU, 'posterior:0':posterior, 'balance_w:0':balance_w}
        self.sess.run(self.activation[5], feed_dict = feed)
        
    def save(self, meta_path, write_model_path):
        meta_graph = tf.train.import_meta_graph(meta_path + '.meta')
        save_path = meta_graph.save(self.sess, write_model_path)
        print('Model is saved in path: %s' % save_path)
        
    def close(self):
        self.sess.close()
        
class ImportRightAU_BP4D():	        
    def __init__(self, model_path):	        
        # Create local graph and use it in the session	        
        self.graph = tf.Graph()	        
        self.sess = tf.Session(graph=self.graph)	        
        with self.graph.as_default():	            
            # Import saved model from location 'loc' into local graph	            
            meta_graph = tf.train.import_meta_graph(model_path + '.meta')	     	                                                                                                    
            meta_graph.restore(self.sess, model_path)	 
            print('Model Restored')	              
            # FROM SAVED COLLECTION:            	            
            self.activation = tf.get_collection('activation') #[p_AUs, pred_AUs, loss_AU_p1, train_AU_p1, loss_AU_p2, train_AU_p2]

    def run(self, image, AUprob, AUconfig):	        
        feed = {'x_image_orig:0': image,   'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig}
        p_AU, pred_AU, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[2]], feed_dict = feed)
        return p_AU, pred_AU, loss

    def run_p2(self, image, AUprob, AUconfig, PGM_p_AU, posterior, balance_w):	        
        feed = {'x_image_orig:0': image,   'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig, 'PGM_p_AUconfig:0': PGM_p_AU, 'posterior:0':posterior, 'balance_w:0':balance_w}
        p_AU, pred_AU, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[4]], feed_dict = feed)
        return p_AU, pred_AU, loss
    
    def train(self, image, AUprob, learning_rate, AUconfig):	    
        feed = {'x_image_orig:0': image,   'learning_rate:0': learning_rate, 'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig}
        self.sess.run(self.activation[3], feed_dict = feed)

    def train_p2(self, image, AUprob, learning_rate, AUconfig, PGM_p_AU, posterior, balance_w):	    
        feed = {'x_image_orig:0': image,   'learning_rate:0': learning_rate, 'label_p_AUconfig:0': AUprob, 'list_AUconfig:0': AUconfig, 'PGM_p_AUconfig:0': PGM_p_AU, 'posterior:0':posterior, 'balance_w:0':balance_w}
        self.sess.run(self.activation[5], feed_dict = feed)
        
    def save(self, meta_path, write_model_path):
        meta_graph = tf.train.import_meta_graph(meta_path + '.meta')
        save_path = meta_graph.save(self.sess, write_model_path)
        print('Model is saved in path: %s' % save_path)
        self.sess.close()


class ImportRightExp():	    
  
    def __init__(self, model_path):	        
        # Create local graph and use it in the session	        
        self.graph = tf.Graph()	        
        self.sess = tf.Session(graph=self.graph)	        
        with self.graph.as_default():	            
            # Import saved model from location 'loc' into local graph
            imported_graph = tf.train.import_meta_graph(model_path + '.meta')	     	                                                                                                    
            imported_graph.restore(self.sess, model_path)	 
            print('Model Restored')	              
            # FROM SAVED COLLECTION:            	            
            self.activation = tf.get_collection('activation') #[p_Exp_K, pred_Exp_K, p_Exp_3rdModel, pred_Exp_3rd, loss_3rd, train_3rd]     

    def run(self, p_au, explabel, PGM_pExp):	        
        feed = {'p_AUs_fix:0' : p_au, 'keep_prob:0': 1, 'label_Expression:0': explabel, 'PGM_p_Exp:0': PGM_pExp}
        #activation 
        p_K, pred_K, p_3rd, pred_3rd, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[2], self.activation[3], self.activation[4]], feed_dict=feed)
        return p_K, pred_K, p_3rd, pred_3rd, loss


    def train(self, p_au, explabel, LR, PGM_pExp):
        feed = {'p_AUs_fix:0' : p_au, 'keep_prob:0': 0.5, 'label_Expression:0': explabel, 'PGM_p_Exp:0': PGM_pExp, 'learning_rate:0':LR}
        #activation [loss_3rd, train_3rd, pred_Exp_3rd, pred_Exp_K]
        self.sess.run(self.activation, feed_dict=feed)
            
        
    def save(self, meta_path, write_model_path):
        meta_graph = tf.train.import_meta_graph(meta_path + '.meta')
        save_path = meta_graph.save(self.sess, write_model_path)
        print('Model is saved in path: %s' % save_path)
        self.sess.close()
class ImportRightExp_BP4D():	    
  
    def __init__(self, model_path):	        
        # Create local graph and use it in the session	        
        self.graph = tf.Graph()	        
        self.sess = tf.Session(graph=self.graph)	        
        with self.graph.as_default():	            
            # Import saved model from location 'loc' into local graph
            imported_graph = tf.train.import_meta_graph(model_path + '.meta')	     	                                                                                                    
            imported_graph.restore(self.sess, model_path)	 
            print('Model Restored')	              
            # FROM SAVED COLLECTION:            	            
            self.activation = tf.get_collection('activation') #[p_Exp_K, pred_Exp_K, p_Exp_3rdModel, pred_Exp_3rd, loss_3rd, train_3rd]     

    def run(self, p_au, explabel, PGM_pExp):	        
        feed = {'p_AUs_fix:0' : p_au, 'label_Expression:0': explabel, 'PGM_p_Exp:0': PGM_pExp}
        #activation 
        p_K, pred_K, p_3rd, pred_3rd, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[2], self.activation[3], self.activation[4]], feed_dict=feed)
        return p_K, pred_K, p_3rd, pred_3rd, loss


    def train(self, p_au, explabel, LR, PGM_pExp):
        feed = {'p_AUs_fix:0' : p_au,  'label_Expression:0': explabel, 'PGM_p_Exp:0': PGM_pExp, 'learning_rate:0':LR}
        #activation [loss_3rd, train_3rd, pred_Exp_3rd, pred_Exp_K]
        self.sess.run(self.activation, feed_dict=feed)
            
        
    def save(self, meta_path, write_model_path):
        meta_graph = tf.train.import_meta_graph(meta_path + '.meta')
        save_path = meta_graph.save(self.sess, write_model_path)
        print('Model is saved in path: %s' % save_path)
        self.sess.close()
        
        
class ImportLeftExp():	    
   
    def __init__(self, loc):	        
        # Create local graph and use it in the session	        
        self.graph = tf.Graph()	        
        self.sess = tf.Session(graph=self.graph)	        
        with self.graph.as_default():	            
            # Import saved model from location 'loc' into local graph	            
            saver = tf.train.import_meta_graph(loc + '.meta')	                                                                                                          
            saver.restore(self.sess, loc)	 
            print('Model Restored')	              
            # FROM SAVED COLLECTION:            	            
            self.activation = tf.get_collection('activation') #p_Expression, pred_Expression, loss_exp_p1, train_Exp_p1, loss_exp_p2, train_exp_p2         
#            print(self.activation)

    def run(self, image, explabel):	        
        
        feed = {'x_image_orig:0': image, 'label_Expression:0': explabel, 'keep_prob:0': 1}
        p_exp, pred_exp, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[2]], feed_dict = feed)
        return p_exp, pred_exp, loss
    
    
    def run_p2(self, image, explabel, posterior, balance_e):
        feed = {'x_image_orig:0': image, 'label_Expression:0': explabel, 'posterior:0': posterior, 'balance_e:0': balance_e, 'keep_prob:0': 1}
        p_exp, pred_exp, loss = self.sess.run([self.activation[0], self.activation[1], self.activation[4]], feed_dict = feed) 
        return p_exp, pred_exp, loss
        
    
    def train(self, image, explabel, LR):
        feed = {'x_image_orig:0': image, 'label_Expression:0': explabel, 'keep_prob:0': 0.5, 'learning_rate:0': LR}
        self.sess.run([self.activation[2], self.activation[3]], feed_dict = feed)
        
        
    def train_p2(self, image, explabel, posterior, balance_e, LR):
        feed = {'x_image_orig:0': image, 'label_Expression:0': explabel, 'posterior:0': posterior, 'balance_e:0': balance_e, 'keep_prob:0': 1, 'learning_rate:0': LR}
        self.sess.run([self.activation[4], self.activation[5]], feed_dict = feed)   
        
        
    def save(self, meta_path, write_model_path):
        meta_graph = tf.train.import_meta_graph(meta_path + '.meta')
        save_path = meta_graph.save(self.sess, write_model_path)
        print('Model is saved in path: %s' % save_path)
        self.sess.close()
