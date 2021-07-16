#coding:utf-8
#!/usr/bin/env python
import numpy as np
import os
from scipy import stats
import argparse

def drop_ousider(data):
    data = np.abs(data)
    data = np.reshape(data,(-1,1024))
    for i in range(data.shape[0]):
        data[i][np.abs(data[i])>0.079] = np.mean(np.abs(data[i][np.abs(data[i]<0.079)]))
    return data

def log_normalization(data, C = -1024.0):
    log_data = np.log(abs(data))
    logsum_data = np.sum(log_data, axis=-1) * (1.0/C)
    logsum_data = np.expand_dims(logsum_data,axis = -1)
    nor_data = log_data / logsum_data
    final_data = np.exp(nor_data)
    return final_data

def KL_divergence_guassian(mean1,mean2,std1,std2):
    return np.log(std2/std1)+ (pow(std1,2)+pow((mean1-mean2),2))/(2*pow(std2,2))-0.5

def JS_divergence_guassian(mean1,mean2,std1,std2):
    return 0.5*(np.log((np.power(std1,2)+np.power(std2,2))/(4*std1*std2)) + 1)

def JS_lowbound(mean1,mean2,std1,std2):
    return 1.0/2*(np.log(2/(1+np.power(np.e,-KL_divergence_guassian(0,0,std1,std2))))+np.log(2/(1+np.power(np.e,-KL_divergence_guassian(0,0,std2,std1)))))

def JS_approximate(mean1,mean2,cov1,cov2):
    sample_num =10
    P = stats.multivariate_normal(mean=mean1, cov=cov1)
    Q = stats.multivariate_normal(mean=mean2, cov=cov2)

    a = np.linspace(-3*cov1,3*cov1,num=sample_num)
    b = np.linspace(-3*cov2,3*cov2,num=sample_num)

    sum = 0
    sum2 = 0
    for i in range(len(a)):
        sum += np.log(P.pdf(a[i])/(1/2*P.pdf(a[i])+1/2*Q.pdf(a[i])))
        sum2 += np.log(Q.pdf(b[i])/(1/2*P.pdf(b[i])+1/2*Q.pdf(b[i])))
    return 1/2*(sum + sum2)/sample_num

def KL_between_two_angle(data1,data2):
    assert data1.shape == data2.shape
    res = 0.0
    for i in range(data1.shape[0]):
        a = JS_lowbound(0,0,data1[i],data2[i])
        res += a
    return res

def correlation_of_sample(data):
    corre_res = 0.0
    Num = data.shape[0]
    for i in range(Num):
        for j in range(i+1,Num):
            a= KL_between_two_angle(data[i], data[j])
            corre_res += a
    return corre_res/(Num * (Num-1)/2 )

def calculate_jsd(data1):
    number = data1.shape[0]
    data1 = log_normalization(data1)
    JS_result = np.zeros(number)
    for i in range(number):
        JS_result[i]=correlation_of_sample(abs(data1[i]))
        print(JS_result[i])
    return JS_result

def JSD(args):
    filepath = args.npz_file
    save_path = args.save_dir
    sample_num = args.sample_num
    model_name = filepath.split('/')[-1].split('_')[0] + '_' + filepath.split('/')[-1].split('_')[1]
    dataset_name = filepath.split('/')[-1].split('_')[-1]
    sigma_final = []

    for i in range(0, sample_num):
        sigma_v = []
        for k in range(0, 40):
            sigma_path = os.path.join(filepath, str(i)+'_'+str(k)+'.npz')
            sigma_val = np.load(sigma_path, allow_pickle=True)['sigma']
            sigma = sigma_val[-2,...]
            if sigma.reshape(1,-1).shape[1]==1:
                sigma = sigma_v[-1]
                print('NoneType Sigma Has been reassigned!')
            sigma_v.append(sigma)
        sigma_final.append(sigma_v)
    sigma_final = np.abs(np.array(sigma_final))
    sigma_final = sigma_final.reshape((sample_num, 40, 1024))

    sigma_final = drop_ousider(sigma_final).reshape(((sample_num, 40, 1024)))
    print(np.max(sigma_final))
    jsd_result = calculate_jsd(sigma_final)
    np.savez(os.path.join(save_path, model_name+'_'+dataset_name+'_jsd.npz'), jsd = jsd_result)
    jsd_mean = np.mean(jsd_result)
    jsd_std = np.std(jsd_result)
    print('*Model:', model_name, '*DataSet:', dataset_name, '*JSD-Mean:', jsd_mean, '*JSD-STD:', jsd_std)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_file', type=str, default='')
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./result')
    args = parser.parse_args()
    JSD(args)
