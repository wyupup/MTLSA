
import numpy as np

class_num=5
batch_size= 100

def convert_to_onehot(sca_label, class_num=class_num):  #sca_label为扫描类别标签      
    return np.eye(class_num)[sca_label]   

class Weight:
#.dtype(int)
    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=batch_size, class_num=class_num): 
        
        batch_size = s_label.size()[0]
        
        s_sca_label = s_label.cpu().data.numpy()#数据转到cpu上进行，然后把tensor变成numpy的数据             
        s_vec_label = convert_to_onehot(s_sca_label)                
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num) #axis=0,按列相加              
        s_sum[s_sum == 0] = 100        
        s_vec_label = s_vec_label / s_sum
        
        
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum
        
        weight_ss = np.zeros((batch_size, batch_size)) #返回来一个给定形状和类型的用0填充的数组
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label) #set() 函数创建一个无序不重复元素集
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1) #-1表示列数自动计算
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                
                ss = np.dot(s_tvec, s_tvec.T) #内积运算
                weight_ss = weight_ss + ss # / np.sum(s_tvec) / np.sum(s_tvec)
                
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt # / np.sum(t_tvec) / np.sum(t_tvec)
                
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st # / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
    
        