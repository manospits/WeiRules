import numpy as np
from torch import nn,from_numpy
import torch.nn.functional as F
import torch
from torch.utils import data
from sklearn.utils import class_weight
from sklearn import tree
from sklearn.tree import _tree
from sklearn.metrics import classification_report
import sys
import copy
from torch import autograd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import export_text

def best_tree(x,y,s,e, criterion,class_val=None):
    best_n=s
    best_s=0
    for n in range(s,e):
        skf = StratifiedKFold(n_splits=5,random_state=0, shuffle=True)
        scores=[]
        for train_index, test_index in skf.split(x, y):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf=tree.DecisionTreeClassifier(criterion=criterion, max_depth=n)
            clf = clf.fit(X_train, y_train)
            preds=clf.predict(X_test)
            if not class_val == None:
                scores.append(classification_report(y_test, preds,output_dict=True)[str(class_val)]['f1-score'])
            else:
                scores.append(classification_report(y_test, preds,output_dict=True)['macro avg']['f1-score'])
        avg=sum(scores)/len(scores)
        if avg > best_s:
            best_s=avg
            best_n=n
    return best_n

class minmax(nn.Module):
    def __init__(self, input_dim, output_dim, rho):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rho = rho
        self.weight = torch.nn.Parameter(torch.ones(output_dim, input_dim))
        self.register_parameter('bias', None)

    def forward(self, input):        
        _, y = input.shape
        if y != self.input_dim:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.input_dim} Input Features')
        
        x = torch.exp(self.rho*input)
        w_normalized = F.softmax(self.weight,dim=1)
        x = F.linear(x,w_normalized)
        x = (1/self.rho)*torch.log(x+1e-8)
        return x
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_dim, self.output_dim, self.bias is not None
        )
    
class Network(nn.Module):
    def __init__(self,
                 slopes_len,
                 class_num,
                 weirules_inst,
                 rulesets,
                 classes,                 
                 rho=14):
        super().__init__()

        self.class_num = class_num
        self.weirules_inst = weirules_inst
        self.slopes_len=slopes_len
        self.rulesets=rulesets
        self.classes=classes
        self.rho=14
        
        flatten_size=1024
        med_size_2 = 512

        #--------------------------------------------------------------
        # convolutional branch
        self.conv1 = nn.Conv2d(256,256,1)
        self.conv2 = nn.Conv2d(256,256,1)
        self.dropout2d_1 = nn.Dropout2d(p=0.2)
        self.batchnorm2d1 = nn.BatchNorm2d(256)
        self.batchnorm2d2 = nn.BatchNorm2d(256)
        self.c_hidden_1 = nn.Linear(12544, flatten_size)
        self.batchnorm1d_c1 = nn.BatchNorm1d(flatten_size)

        # merge
        self.layer1 = nn.Linear(flatten_size, med_size_2)
        self.batchnorm1d_4 = nn.BatchNorm1d(med_size_2)
        self.merge_hidden = nn.Linear(med_size_2, self.slopes_len)
        self.ors= nn.ModuleList([minmax(len(self.weirules_inst.rulesets[c]),1,self.rho) for c in self.weirules_inst.en_classes])

        self.relu = F.relu
        self.lrelu = nn.LeakyReLU()
        

    def forward(self, deep_input, rule_input=None):
        
        if rule_input==None:
            rule_input=deep_input
                
        
        # convolutional input
        cx = self.conv1(deep_input)
        cx = self.batchnorm2d1(cx)
        cx = self.relu(cx)
        cx = self.conv2(cx)
        cx = self.batchnorm2d2(cx)
        cx = self.relu(cx)
        
        cx = cx.flatten(start_dim=1)
        cx = self.c_hidden_1(cx)
        cx = self.batchnorm1d_c1(cx)
        cx = self.relu(cx)

        #non conv
        x = self.layer1(cx)
        x = self.batchnorm1d_4(x)
        x = self.relu(x)
        
        o2 = self.merge_hidden(x)
        o2 = self.lrelu(o2)
        ruleset_results_batch = self.weirules_inst.compute_ruleset_vector(rule_input,o2)
                    
        all_softmax_results = self.rule_inference(ruleset_results_batch, self.classes, self.rulesets, ors=self.ors,rho=self.rho)    
        return all_softmax_results

    
    def rule_inference(self,ruleset_results_batch, classes, rulesets, ors, rho=14):
        rsr_s=0
        and_rho=-rho
        class_rules_results=[]
        #loop over class cases
        c=0
        for class_name in classes:
            class_ruleset_lens = [len(x) for x in rulesets[class_name]]
            ruleset_offset = sum(class_ruleset_lens)
            ruleset_results=ruleset_results_batch[:,rsr_s:rsr_s+ruleset_offset] #results from current rule
            rw_s=0
            #loop over conditions of rule
            ruleset_anded_results=[]
            for number_of_comps in class_ruleset_lens:
                rule_results_tensor=ruleset_results[:,rw_s:rw_s+number_of_comps]
                #------- compute and operation for comparisons of this condition ---------
                N=rule_results_tensor.shape[1]
                rule_anded_result=torch.clamp(weighted_exponential_mean(rule_results_tensor, N, and_rho), min = 0, max = 1)
                ruleset_anded_results.append(rule_anded_result)

                # advance weight index
                rw_s+=number_of_comps

            #create tensor containing results of all conditions
            all_rules=torch.stack(ruleset_anded_results,dim=1)
            #------ or operation among all rules -------
            N = all_rules.shape[1]
            or_val = ors[c](all_rules).reshape(-1)

            class_rules_results.append(or_val)
            rsr_s+=ruleset_offset
            c+=1

        class_rule_values=torch.stack(class_rules_results, dim=1)
        return class_rule_values

    

    def cross_entropy(self, w, l):
        #normalize
        if self.weirules_inst.use_weights==True:
            tensor_class_weights=torch.tensor(self.weirules_inst.class_weights).float().to(self.weirules_inst.device)
            return F.nll_loss(torch.log(F.normalize(w, p=1 ,dim=1)+1e-8), l, weight=tensor_class_weights)
        else:
            return F.nll_loss(torch.log(F.normalize(w, p=1 ,dim=1)+1e-8), l)

class weirules():

    def __init__(self, rule_learner='tree', use_weights=False):
        self.en_classes=[]
        self.rulesets={}
        self.all_comps={}
        self.mapping=None
        self.model=None
        self.rule_learner=rule_learner
        self.use_weights=use_weights
        self.rule_cols_index=None
        self.df_cols_index=None
        self.rule_clf=None
        self.rule_lens=[]
        use_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def column_map(self,name):
        return self.mapping[name] #index

    def sigmf(self,x,a,c):
        val = torch.sigmoid(a * (x - c))
        return val
    
    def compute_ruleset_vector(self,X,slopes=None):
        result_vector_len=0
        for class_value in self.en_classes:
            classModel=self.rulesets[class_value]
            for rule in classModel:
                result_vector_len+=len(rule)
        all_weighted_results=torch.zeros([len(X),result_vector_len]).to(self.device)
        t_index=0
        s_index=0
        for class_value in self.en_classes:
            comp_index=0
            for comp in self.all_comps[class_value]:
                [feature,op,value]=comp
                col_index=self.column_map(feature)
                if op == '<=':
                    all_weighted_results[:,t_index+comp_index]=self.sigmf(-X[:,col_index],slopes[:,s_index+comp_index],-float(value))
                elif op == '>':
                    all_weighted_results[:,t_index+comp_index]=self.sigmf(X[:,col_index],slopes[:,s_index+comp_index],float(value))
                comp_index+=1
            s_index+=comp_index
            t_index+=comp_index
        return all_weighted_results
   

    def fit_tree(self,
                 X,                      # train X DATA
                 Y,                      # train Class data
                 en_classes,             # list of the classes
                 rule_columns,           # list containing the feature names
                 max_depth=None,         # max depth for dt training
                 criterion='gini',       # dt split criterion
                 find_best_tree=False,
                 forest=False
                ):
        
        X_rules = X[rule_columns]
        self.rule_cols_index = [X.columns.get_loc(c) for c in rule_columns]
        self.mapping = dict([(X_rules.columns[i],i) for i in range(len(list(X_rules.columns)))])
        self.mapping = dict([(X_rules.columns[i],i) for i in range(len(list(X_rules.columns)))])

        self.en_classes=en_classes
        self.class_weights=None
        if self.use_weights:
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes= en_classes,
                                                 y=Y)
        self.slope_vector_length = 0
        
        rulesets={}
        if find_best_tree:
            if max_depth==None:
                print("Warning: max_depth was set to 30.")
                max_depth=30
            if not forest==True:
                Y_cls=Y.copy()
                best_depth = best_tree(X_rules, Y_cls,2, max_depth,criterion)
                clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=best_depth)
                clf = clf.fit(X_rules, Y_cls)
                self.clf=clf
                treecode=tree_to_rules(clf,X_rules.columns)
                for class_value in self.en_classes:
                    rulesets[class_value]=treecode[class_value]
            else:
                for class_value in self.en_classes:
                    Y_cls=Y.copy()
                    Y_cls[Y_cls!=class_value]=-1
                    best_depth = best_tree(X_rules, Y_cls,2, max_depth,criterion,class_value)
                    #print(best_depth)
                    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=best_depth)
                    clf = clf.fit(X_rules, Y_cls)
                    rulesets[class_value]=tree_to_rules(clf,X_rules.columns)[class_value]
        else:
            if not forest==True:
                Y_cls=Y.copy()
                clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
                clf = clf.fit(X_rules, Y_cls)
                self.clf=clf
                treecode=tree_to_rules(clf,X_rules.columns)
                for class_value in self.en_classes:
                    rulesets[class_value]=treecode[class_value]
            else:
                for class_value in self.en_classes:
                    Y_cls=Y.copy()
                    Y_cls[Y_cls!=class_value]=-1
                    clf = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
                    clf = clf.fit(X_rules, Y_cls)
                    rulesets[class_value]=tree_to_rules(clf,X_rules.columns)[class_value]              
            
        self.rulesets=rulesets
        
        for class_value in self.en_classes:
            classModel=self.rulesets[class_value]
            cond_lens=[]
            for cond in classModel:
                self.slope_vector_length+=len(cond)
                cond_lens.append(len(cond))
                for comp in cond:
                    if not class_value in self.all_comps:
                        self.all_comps[class_value]=[comp]
                    else:
                        self.all_comps[class_value].append(comp)
            self.rule_lens.append(cond_lens)
    
    
    def create_network(self, rho=14):
        self.model = Network(slopes_len = self.slope_vector_length,    
                         class_num=len(self.en_classes),
                         weirules_inst=self,
                         rulesets=self.rulesets,
                         classes=self.en_classes,
                         rho=rho)
        
        self.model.float()
        self.model.to(self.device)

    def forward_model(self, Xdf, Xrule):
        all_softmax_results = self.model.forward(Xdf, Xrule)
        return all_softmax_results
    
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    
def weighted_exponential_mean(X, N, rho, W = None):
    if (W == None):
        return (1/rho)*torch.log((1/N) *torch.sum(torch.exp(rho*X),dim=1)+1e-8)
    else:
        return (1/rho)*torch.log(torch.sum(W*torch.exp(rho*X),dim=1)+1e-8)


def tree_to_rules(tree, feature_names):
    #tree_rules = export_text(tree)
    #print(tree_rules)
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    classes = tree.classes_
    def recurse(node, rule_list, rulesets):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            rule_list_l=rule_list+[[name, '<=',threshold]]            
            recurse(tree_.children_left[node],rule_list_l,rulesets)
            rule_list_r=rule_list+[[name, '>',threshold]]           
            recurse(tree_.children_right[node],rule_list_r,rulesets)
        else:
            class_value=classes[np.argmax(tree_.value[node])]
            if class_value in rulesets:
                rulesets[class_value].append(rule_list)
            else:
                rulesets[class_value]=[rule_list]
                

    rulesets={}
    recurse(0, [], rulesets)
    return rulesets

