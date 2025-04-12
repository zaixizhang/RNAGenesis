import os
import sys
import re
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import copy
import torch
import train_val
import reader
import logomaker
from torch import nn
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from importlib import reload
from scipy import stats
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.manifold import TSNE
import ruptures as rpt
from popen import Auto_popen

tensor_2_numpy = lambda x: x.detach().cpu().numpy()

class Maximum_activation_patch(object):
    def __init__(self, popen, which_layer, n_patch=9, kfold_index=None, device_string='cpu'):
        self.popen = popen
        self.layer = which_layer
        self.popen.cuda_id = torch.device("cuda:%s"%device_string) if device_string.isdigit() else torch.device('cpu')
        self.n_patch = n_patch
        self.kfold_index = kfold_index
        self.total_stride = np.product(popen.stride[:self.layer])
        self.compute_reception_field()
        self.compute_virtual_pad()
    
    def load_indexed_dataloader(self,task):
        #        True or 'train_val'
        assert (self.popen.kfold_cv!=False) == (self.kfold_index != None), \
                "kfold CV should match with kfold index"
        
        self.popen.kfold_index = self.kfold_index
        self.popen.shuffle = False
        self.popen.pad_to = 57 if self.popen.cycle_set==None else 105
        
        tmp_popen = copy.copy(self.popen)
        if self.popen.cycle_set != None:
            base_path = copy.copy(self.popen.split_like)
            base_csv = copy.copy(self.popen.csv_path)
            
            if base_path is not None:
                assert task != None , "task is not defined !"
                tmp_popen.split_like = [path.replace('cycle', task) for path in base_path]
            else:
                tmp_popen.csv_path = base_csv.replace('cycle', task)
                
        return  reader.get_dataloader(tmp_popen) 
    
    def load_model(self):
        # if self.kfold_index is not None:
        #     base_pth = self.popen.vae_pth_path
        #     # self.popen.vae_pth_path = base_pth.replace(".pth","_cv%s.pth"%self.kfold_index)
        self.popen.kfold_index = self.kfold_index
        model = utils.load_model(self.popen, None)
        # if self.kfold_index is not None:
        #     self.popen.vae_pth_path = base_pth
        return model
    
    def get_filter_param(self, model):
        Conv_layer = model.soft_share.encoder[self.layer-1]
        
        return tensor_2_numpy( next(Conv_layer[0][0].parameters()) )
    
    def loading(self,task, which_set):
        model = self.load_model().to(self.popen.cuda_id)
        dataloader = self.load_indexed_dataloader(task)[which_set]
        self.df = dataloader.dataset.df
        return model, dataloader
    
    @torch.no_grad()
    def cumulative_rl_decision(self, task=None,which_set=0, extra_loader=None):
        """
        take out the memory h_i of each position and pass to output layer
        Arg:
            task : str
            which_set : int, 0 : training set, 1 : val set, 2 : test set
        """
        model, dataloader= self.loading(task, which_set)
        model.eval()
        
        Y_ls = []
        for Data in tqdm(dataloader):
            # iter each batch
            x,y = train_val.put_data_to_cuda(Data,self.popen,False)
            x = torch.transpose(x, 1, 2)
            y_pred = model.predict_each_position(x)
            Y_ls.append( tensor_2_numpy(y_pred) )
        Y_ay = np.concatenate(Y_ls, axis=0)
        
        # release some cache
        torch.cuda.empty_cache()
        del model
        return Y_ay.reshape(Y_ay.shape[0],-1)
        
    def extract_feature_map(self, task=None,which_set=0, extra_loader=None):
        """
        load trained model and unshuffled dataloader, make model forwarded
        Arg:
            task : str
            which_set : int, 0 : training set, 1 : val set, 2 : test set
        """
        model, dataloader= self.loading(task, which_set)
        if extra_loader is not None:
            dataloader = extra_loader
            self.df = extra_loader.dataset.df
            
        feature_map = []
        X_ls = []
        Y_ls = []
        
        model.eval()
        with torch.no_grad():
            for Data in tqdm(dataloader):
                # iter each batch
                x,y = train_val.put_data_to_cuda(Data,self.popen,False)
                x = torch.transpose(x, 1, 2)
#                 X_ls.append(x.numpy())
                Y_ls.append( tensor_2_numpy(y))
                
                for layer in model.soft_share.encoder[:self.layer]:
                    out = layer(x)
                    x = out
                feature_map.append( tensor_2_numpy(out))
                
                torch.cuda.empty_cache()
            

        feature_map_l = np.concatenate( feature_map, axis=0)
        
#         self.X_ls = np.concatenate(X_ls, axis=0)
        self.Y_ls = np.concatenate(Y_ls, axis=0)

        print("activation map of layer |%d|"%self.layer,feature_map_l.shape)
#         print(self.X_ls.shape)
        print("Y : ",self.Y_ls.shape)
        self.feature_map = feature_map_l
        
        self.filters = self.get_filter_param(model)
        
        del model
        
        return feature_map_l
    
    def compute_reception_field(self):
        r = 1
        strides = self.popen.stride[:self.layer][::-1]

        for i,s in enumerate(strides):
            r = s*(r-1) + self.popen.kernel_size
        
        self.r = r
        self.strides = strides
        print('the reception filed is ', r)
    
    def compute_virtual_pad(self):
        v = []
        pd = self.popen.padding_ls[:self.layer]

        for i,p in enumerate(pd):
            v.append(p*np.product(self.popen.stride[:len(pd)-i-1]))
        
        self.virtual_pad = np.sum(v)
        print('the virtual pad is', self.virtual_pad)
    
    def retrieve_input_site(self, high_layer_site):
        """pad ?"""
        
        virtual_start = high_layer_site*self.total_stride - self.virtual_pad
        start = max(0, virtual_start)
        
        end = virtual_start + self.r
        return max(0,int(start)), int(end)
    
    def detect_changepoint(self,time_series):
        bkpt = rpt.KernelCPD(kernel="linear", min_size=1).fit_predict(time_series, n_bkps=1)[0]
        return bkpt - 1        
    
    def retrieve_featmap_at_changepoint(self, featmap, rl_chain, threshold=1, direction='less', detect_region=None):
        """
        Using the change point of rl series, to retrieve feature map at the same position
        e.g. : to find negative change point threshold=-1, direction='less'
        e.g. : to find positive change point threshold=0.5, direction='greater'
        
        featmap : np.ndarray, (n_sample, n_channel, n_position)
        rl_chain : np.ndarray, (n_sample,  n_position)

        threshold : the threshold define the rl after change point minus that ahead the point that are considered
        direction : the direction of changes
        detect_region : list of slice [], which region of the rl chain is used to detect change point, default None (full sequence is considered)
        """
        # take out the negative break point
        
        if direction == 'less':
            condition = lambda x1, x2 : x1 - x2 < threshold
        elif direction == 'greater':
            condition = lambda x1, x2 : x1 - x2 > threshold

        if detect_region is None:
            detect_region = [slice(0, None)] * rl_chain.shape[0]

        change_point_act = []
        for i, trend in enumerate(rl_chain):
            region = detect_region[i]
            bkpt= self.detect_changepoint(trend[region]) + region.start

            if condition(trend[bkpt+1] , trend[bkpt]): 
                activation_vec = featmap[i, :, bkpt+1]
                change_point_act.append(activation_vec)

        chagnepoint_map = np.asarray(change_point_act)
        return chagnepoint_map
    
    def locate_MA_seq(self, channel, feature_map=None):
        """
        
        """
        if feature_map is None:
            feature_map = self.feature_map
            
        channel_feature = feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)

        max_n_index = np.argpartition(F0_ay, -1*self.n_patch, axis=0)[-1*self.n_patch:]


        # a patch of sequences
        max_patch = self.df[self.popen.seq_col].values[max_n_index]

        # find the location of maximally acitvated
        Conv4_sites = np.argmax(channel_feature[max_n_index],axis=1)
        
        mapped_input_sites = [self.retrieve_input_site(site) for site in Conv4_sites]
        
        # the mapped region of the sequences
        max_act_region = []
        for utr, (start, end) in zip(max_patch, mapped_input_sites):
            

            pad_gap = self.popen.pad_to - len(utr)
            field = utr[max(0, start-pad_gap): end-pad_gap]
            
            max_act_region.append(field)
            
#         print(mapped_input_sites)
        
        full_field = [len(field)==self.r for field in max_act_region]
        
        return np.array(max_act_region)[full_field], max_n_index[full_field], np.array(mapped_input_sites)[full_field]
    
    def sequence_to_matrix(self, max_act_region, weight=None, transformation='counts'):
        
        assert np.all([seq != "" for seq in max_act_region])
        
        max_len = max([len(seq) for seq in max_act_region])
        
        M = np.zeros((max_len,4))
        if weight is None:
            weight = np.ones((self.n_patch,))
        for seq, w in zip(max_act_region, weight):
            oh_M = reader.one_hot(seq)*w
            M += np.concatenate([np.zeros((max_len - len(seq),4)), oh_M],axis=0)
            

        seq_logo_df = pd.DataFrame(M, columns=['A', 'C', 'G', 'U'])
        if transformation!='counts':
            seq_logo_df = logomaker.transform_matrix(seq_logo_df, from_type='counts', to_type=transformation)
            
        return seq_logo_df
        
    def plot_sequence_logo(self, seq_logo_df,  save_fig_path=None, ax=None):
        """
        input : max_act_region ; list of str
        """
        # plot
        if ax is None:
            fig = plt.figure(dpi=300)
            ax = fig.gca()
        MA_C = logomaker.Logo(seq_logo_df,ax=ax);
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # save
        if save_fig_path is not None:
            MA_C.fig.savefig(save_fig_path,transparent=True,dpi=600)
            save_dir = os.path.dirname(save_fig_path)
            
            try:
                self.save_dir
            except:
                # which is the first time we save
                self.save_dir = save_dir
                print('fig saved to',self.save_dir)
            plt.close(MA_C.fig)
    
    def fast_logo(self, channel, feature_map=None, n_patch=None, transformation='information', save_fig_path=None, title=None,ax=None):
        
        if n_patch is not None:
            self.n_patch = n_patch
        max_act_region, _, _ = self.locate_MA_seq(channel, feature_map)
        M = self.sequence_to_matrix(max_act_region, transformation=transformation)
        F0_ay, (spr,pr) = self.activation_density(channel, False, False)
        self.plot_sequence_logo(M,  save_fig_path=save_fig_path, ax=ax)
        if ax is None:
            ax=plt.gca()
        if title is None:
            title = "filter {} : $r =$ {}".format(channel, spr)
        
        ax.set_title(title, fontsize=35)
         
    
    def activation_density(self, channel, to_print=True, to_plot=True, feature_map=None, **kwargs):
        if feature_map is None:
            feature_map = self.feature_map
            
        channel_feature = feature_map[:, channel,:]
        F0_ay = channel_feature.max(axis=-1)
        if to_plot:
            sns.kdeplot(F0_ay, **kwargs);
        if to_print:
            print("num of max acti: %s"%np.sum(F0_ay==F0_ay.max()))
            
            n_gtr = [np.sum(F0_ay > q) for q in np.quantile(F0_ay,[0.5,0.8,0.95])]
            print("quantiles: 50% {} 80% {} 95% {}".format(*n_gtr))
        
        spr = stats.spearmanr(F0_ay, self.Y_ls.flatten())
        pr = stats.pearsonr(F0_ay, self.Y_ls.flatten())
        return F0_ay, (round(spr[0],3),round(pr[0],3))
    
    def within_patch_clustering(self, channel, n_clusters, n_patch=None , to_plot=True,**kwargs):
        """
        return:
            subclusters : list, list of patch (alignments)
        """
        self.n_patch = n_patch
        act_patch, _, _ = self.locate_MA_seq(channel)
        flatten_seq = np.stack([reader.one_hot(seq).flatten() for seq in act_patch])
        print("the shape of sequence matrix {}".format(flatten_seq.shape))
        
        # clsutering
        cluster_index = cluster.KMeans(n_clusters=n_clusters).fit_predict(flatten_seq)
        # down
        if to_plot:
            tsne = TSNE(metric='cosine').fit(flatten_seq)
            self.tsne=tsne
            plt.figure(figsize=(6,5),dpi=150)
    #         sns.set_theme(style='ticks', palette='viridis')
            scatter_args = {"palette":'viridis'}
            scatter_args.update(kwargs)
            sns.scatterplot(x=tsne.embedding_[:,0], y=tsne.embedding_[:,1], 
                            hue=cluster_index, **scatter_args);
        
        return act_patch, flatten_seq, cluster_index
    
    def activation_overview(self, **kwargs):
        channel_spearman = []
       
        fig, ax = plt.subplots(1,1, figsize=(10,5), dpi=300)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        n_channel = self.popen.channel_ls[self.layer]
        for i in tqdm(range(n_channel)):
            act, (spr, pr)  = self.activation_density(i, to_print=False, feature_map=None, **kwargs);
            channel_spearman.append(spr)
        return fig, ax, np.array(channel_spearman)
    
    def matrix_to_seq(self, matrix):
        """
        return the sequence with maximum weight at each site
        """
        max_act_seq = ''
        assert matrix.shape[1] == 4
        for i in matrix.argmax(axis=1):
            max_act_seq += ['A', 'C', 'G', 'T'][i]
            
        return max_act_seq
    
    def save_as_meme_format(self,channels:list, save_path, filter_prefix='filter', transformation='probability'):
        motifs = []
        success_channel = []
        for cc in channels:
            try:
                region, index, patches = self.locate_MA_seq(channel=cc)
                M = self.sequence_to_matrix(region, transformation=transformation);
                motifs.append(M)
                success_channel.append(cc)
            except ValueError:
                continue
        if save_path is None:
            return motifs, success_channel
        else:
            write_meme(success_channel, motifs ,save_path, filter_prefix)
    
    def gradience_scaler(self,array):
        """
        a scaling function to 
        """
        meann = np.mean(array)
        ranges = array.max() -  array.min()
        return (array-meann)/ranges
    
    def get_input_grad(self, task, focus=True, fm=None, starting_layer=None):
        """
        compute the gradience of Y over feature_map
        fm : specify feature map
        starting_layer: or None [0,3]
        """
        All_grad = []
        current_layer = self.layer if starting_layer is None else starting_layer
        model = self.load_model().to(self.popen.cuda_id)
        model.train()
        model.task = task
        if fm is None:
            fm = self.feature_map
        
        for start in tqdm(range(0, fm.shape[0], 64)):
            # prepare input
            minibatch = fm[start:start+64]
            X = torch.as_tensor(minibatch, device=self.popen.cuda_id).float()
            X.requires_grad=True

            # forward
            # x = model.soft_share.encoder[3](X)
            out=X
            for layer in model.soft_share.encoder[current_layer:]:
                out = layer(out)

            out = model.forward_tower(out)

            # auto grad
            external_grad = torch.ones_like(out)
            out.backward(gradient=external_grad,retain_graph=True)
            grad = X.grad
            
            if focus:
                indices = self.argmax_to_indeces(np.argmax(minibatch, axis=2))
                grad = grad[indices].reshape(-1,256)
            
            All_grad.append( tensor_2_numpy(grad) )
        
        # concate each channel and average over input sequences
        grad_ay = np.concatenate(All_grad, axis=0)#.mean(axis=0)

        # return self.gradience_scaler(grad_ay)
        return grad_ay  
            
    def argmax_to_indeces(self,index):
        "index : of shape [batch, 256] , the result of "
        s_index = []
        ch_index = []
        loc_index = []
        for sample in range(index.shape[0]):
            sam_loc = index[sample]
            for ch, pos in enumerate(sam_loc):
                s_index.append(sample)
                ch_index.append(ch)
                loc_index.append(pos)
                
        return (s_index, ch_index, loc_index)

# def extract_max_seq_pattern(condition, n_clusters=6, n_patch=3000):
#     pattern = []
#     channel_source = []
#     for channel in np.where(condition)[0]:

#         try:
#             act_patch, flatten_seq, cluster_index = self.within_patch_clustering(channel, n_clusters=n_clusters, to_plot=False,n_patch=n_patch)
#         except ValueError:
#             continue

#         for i in range(n_clusters):
#             sub_cluster = act_patch[cluster_index==i]
#             if len(sub_cluster) > 0:
#                 matrix = self.sequence_to_matrix(sub_cluster)
#                 pattern.append(self.matrix_to_seq(matrix))
#                 channel_source.append(channel)
#         return pattern, channel_source

def sum_occurrance(df, pattern):
    pattern_occurance = []
    for p in pattern:
        pattern_occurance.append(np.sum([(p in utr) for utr in df.seq.values]))
    return np.array(pattern_occurance)

def generate_scramble_index(size  , N_1):
    scramble_index=np.zeros((size,))

    while scramble_index.sum() < N_1:    
        num_ = int(N_1 - scramble_index.sum())
        randindex = np.random.randint(0, size, size=(num_,))

        for i in randindex:
            scramble_index[i] 
    return scramble_index    


class merge_task_map(Maximum_activation_patch):
    def __init__(self, popen, which_layer, n_patch , kfold_index=None, device_string='cpu'):
        """merging all tasksa"""
        super().__init__(popen, which_layer, n_patch , kfold_index, device_string)
        self.old_popen = copy.copy(popen)
        self.df_dict = {}
        self.Y_2_task = {}
        self.patches_ls = None
    
    def load_indexed_dataloader(self,task):
        
        if task in ['Andrev2015','muscle','pc3']:
            self.popen.seq_col = 'utr'
            self.popen.aux_task_columns = ['log_te']
            self.popen.split_like = None
            self.popen.kfold_cv = True
        
        loader_ls = super().load_indexed_dataloader(task)
        self.popen = copy.copy(self.old_popen)
        return loader_ls
    
    def extract_feature_map(self, which_set=0):
        feature_map = {}
        for task in self.popen.cycle_set:
            feature_map[task] = super().extract_feature_map(task=task, which_set=which_set)
            self.Y_2_task[task] = self.Y_ls
        self.feature_map = feature_map
        return feature_map
    
    def loading(self,task, which_set):
        model = self.load_model().to(self.popen.cuda_id)
        dataloader = self.load_indexed_dataloader(task)[which_set]
        self.df_dict[task] = dataloader.dataset.df
        return model, dataloader
    
    def locate_MA_seq(self, channel, feature_map=None, patches_ls=None):
        """A multi-task version of MA region retrieve"""
        regions = []
        indeces = {}
        patches = []
        
        if patches_ls is None:
            patches_ls = self.patches_ls
        
        for i, task in enumerate(self.popen.cycle_set):
            # task out : region, index, patches
            if patches_ls is not None:
                self.n_patch = patches_ls[i]
                # print(f"{task} n_patch: {self.n_patch}")
            self.df = self.df_dict[task]
            
            r, i, p = super().locate_MA_seq(channel, feature_map=self.feature_map[task]) #region, index, patches
            regions.append(r)
            indeces[task] = i
            patches.append(p)
            
        return np.concatenate(regions) , indeces , np.concatenate(patches)
    
    def activation_density(self, channel, to_print=True, to_plot=True, feature_map=None, **kwargs):
        all_F0 = []
        all_spr = []
        all_pr = []
        for i, task in enumerate(self.popen.cycle_set):
            # task out : region, index, patches
            task_featmap = self.feature_map[task]
            self.Y_ls = self.Y_2_task[task]
            
            if to_print:
                print(task)
            F0, (spr,pr) = super().activation_density(channel, to_print=to_print, 
                                                  to_plot=False,
                                                  feature_map = task_featmap)
            all_F0.append(F0)
            all_spr.append(spr)
            all_pr.append(pr)
        
        # merge all activateion 
        
        if to_plot:
            for F0_ay in all_F0:
                sns.kdeplot(F0_ay, **kwargs)
            
        return all_F0 , (all_spr, all_pr)
    
    def flexible_n_patch(self,channel, qtl=0.95):
        """
        call this function before the all other functions will enable a flexible task-wise n_patch
        
        """
        all_F0 , (all_spr, all_pr) = self.activation_density(channel,False, False,None)
        thres = np.max([np.quantile(f0, qtl) for f0 in all_F0])
        self.patches_ls = [np.sum(f0 > thres) for f0 in all_F0]
        return self.patches_ls 
        
    def activation_overview(self):
        pr_ay=[]
        spr_ay=[]

        # the number of convolution filters in layer 3
        # the index is also 3 because input channel = 4 for layer 0
        for i in tqdm(range(self.popen.channel_ls[self.layer])):
            # 
            _,(spr,pr) = self.activation_density(i,False,False)
            pr_ay.append(pr)
            spr_ay.append(spr)

        # convert to ndarray
        pr_ay = np.stack(pr_ay) 
        spr_ay = np.stack(spr_ay) 
        return pr_ay, spr_ay
    
    def save_as_meme_format(self,channels:list, save_path, filter_prefix='filter', transformation='probability', qtl=0.95, fix_patches_ls=None):
        """
        Save the position weight matrix as the meme-suite acceptale minimal motif format
        """
        
        with open(save_path, 'w') as f:
            f.write("MEME version 5.3.0\n\n")
            f.write("ALPHABET= ACGT\n\n")
            f.write("strands: + -\n\n")
            f.write("Background letter frequencies\n")
            f.write("A 0.25 C 0.25 G 0.25 T 0.25\n")

            for cc in channels:
                if fix_patches_ls is None:
                    n_patches_ls = self.flexible_n_patch(cc, qtl=qtl)
                else:
                    n_patches_ls = fix_patches_ls
                try:
                    region, index, patches = self.locate_MA_seq(channel=cc, patches_ls=n_patches_ls)
                    M = self.sequence_to_matrix(region, transformation=transformation);
                    f.write('\n')
                    f.write(f"MOTIF {filter_prefix}_{cc}\n")
                    seq_len = len(region[0])
                    f.write(f"letter-probability matrix: alength= 4 w= {seq_len} \n")
                    for line in M.values:
                        f.write(" "+line.__str__()[1:-1]+'\n')
                except ValueError:
                    continue
                    
            f.close()
            print('writed')
            
    def get_input_grad(self, focus=True):
        grads = {}
        for task in self.popen.cycle_set:
            grads[task] = super().get_input_grad(focus=True, task=task, fm=self.feature_map[task]) 
        return grads

class Maximum_activation_kmer(Maximum_activation_patch):
    def __init__(self,popen, which_layer, n_patch, kfold_index):
        super().__init__(popen, which_layer, n_patch, kfold_index)
        self.virtual_pad = 0
    
    def compute_virtual_pad(self):
        self.virtual_pad = 0
        print('the virtual pad is', self.virtual_pad)
    
    def load_indexed_dataloader(self, task='kmer'):
        self.csv_path = f"<DATA_DIR>/all_{self.r}mer.csv"
        self.popen.split_like = [self.csv_path,self.csv_path]
        self.popen.csv_path = None
        self.popen.kfold_cv = False
        self.popen.kfold_index = self.kfold_index
        self.popen.pad_to = self.r
        self.popen.seq_col = 'utr'
        self.popen.aux_task_columns = ['rl']
        return  reader.get_dataloader(self.popen) 
        
    def extract_feature_map(self):
         
        model, dataloader= self.loading('kmer', 1)
        
        for l in range(self.layer):
            # remove padding to precisely locate
            model.soft_share.encoder[l][0][0].padding = (0,)
        
        feature_map = []
        X_ls = []
        Y_ls = []
        
        model.eval()
        with torch.no_grad():
            for Data in tqdm(dataloader):
                # iter each batch
                x,y = train_val.put_data_to_cuda(Data,self.popen,False)
                x = torch.transpose(x, 1, 2)
#                 X_ls.append(x.numpy())
                Y_ls.append( tensor_2_numpy(y) )
                
                for layer in model.soft_share.encoder[:self.layer]:
                    out = layer(x)
                    x = out
                feature_map.append( tensor_2_numpy(out) )
                
                torch.cuda.empty_cache()
            

        feature_map_l = np.concatenate( feature_map, axis=0)
        
#         self.X_ls = np.concatenate(X_ls, axis=0)
        self.Y_ls = np.concatenate(Y_ls, axis=0)

        print("activation map of layer |%d|"%self.layer,feature_map_l.shape)
#         print(self.X_ls.shape)
        print("Y : ",self.Y_ls.shape)
        self.feature_map = feature_map_l
        
        self.filters = self.get_filter_param(model)
        
        del model
        self.df = pd.read_csv(self.csv_path)
        return feature_map_l
    
def write_meme(channels:list, PWMs:list ,save_path, filter_prefix='filter'):
    """
    Save the position weight matrix as the meme-suite acceptale minimal motif format
    """
    assert len(channels)==len(PWMs)
    with open(save_path, 'w') as f:
        f.write("MEME version 5.4.1\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n")

        for cc,M in zip(channels, PWMs):
            f.write('\n')
            f.write(f"MOTIF {filter_prefix}_{cc}\n")
            seq_len = M.shape[0]
            f.write(f"letter-probability matrix: alength= 4 w= {seq_len} \n")
            for line in M.values:
                f.write(" "+line.__str__()[1:-1]+'\n')


        f.close()
        print('writed to', save_path)
        
def extract_meme(memepath):
    """
    read the meme file and extract motifs to rewrite
    """
    with open(memepath,'r') as f:
        all_lines = f.readlines()[9:]
    
    all_blocks = []
    for i, line in enumerate(all_lines):
        if line.startswith("MOTIF"):
            width = re.match(r"letter-probability matrix: alength= 4 w= (\d)*",all_lines[i+1]).groups(1)
            width = int(width[0])

            all_blocks.append( all_lines[i:i+3+width])
            
    return all_blocks

def PWM_2_consensus(df):
    base = ['A','C','G','T']
    consensus = ''
    for vec in df.values:
        consensus += base[vec.argmax()]
    return consensus

def read_tomtom_RBP_out(tomtom_out_path, RBP_meme=None):
    """
    tomtom_out_path : str, the path of tomtom outcome, tsv format
    RBP_meme : str, the meme-suite built-in motifs database, Ray-2013 human
    """
    # the last 3 lines are 
    RBP_df = pd.read_table(tomtom_out_path).iloc[:-3]
    if RBP_meme is None:
        RBP_meme = "/ssd/users/wergillius/tool/meme/motif_databases/RNA/Ray2013_rbp_Homo_sapiens.dna_encoded.meme"

    # align the gene symbol to the query motif id
    with open(RBP_meme, "r") as f:
        lines = f.readlines()
        motif_rbp_dict = {}
        for line in lines:
            if 'MOTIF' in line:
                _, RBP_id, Gene_Symbol = line.strip().split()
                motif_rbp_dict[RBP_id] = Gene_Symbol
        f.close()

    RBP_df['Gene_Symbol'] = RBP_df['Query_ID'].map(motif_rbp_dict)
    RBP_df['filters'] = RBP_df['Target_ID'].apply(lambda x: int(x.split("_")[1]))
    return RBP_df