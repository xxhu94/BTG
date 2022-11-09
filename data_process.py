
import pandas as pd
from scipy.stats import entropy, pearsonr, stats
from scipy.stats import norm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import sparse

path = r'./data/'
user_data = r"./data/user_data/"


def get_app_feats(df):
    print(df.head())
    print(df["busi_name"].value_counts())
    phones_app = df[["phone_no_m"]].copy()
    phones_app = phones_app.drop_duplicates(subset=['phone_no_m'], keep='last')
    tmp = df.groupby("phone_no_m")["busi_name"].agg(busi_count="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["flow"].agg(flow_mean="mean",
                                               flow_median="median",
                                               flow_min="min",
                                               flow_max="max",
                                               flow_var="var",
                                               flow_sum="sum")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["month_id"].agg(month_ids="nunique")
    phones_app = phones_app.merge(tmp, on="phone_no_m", how="left")
    phones_app["flow_month"] = phones_app["flow_sum"] / phones_app["month_ids"]

    return phones_app


def get_voc_feat(df):
    df["start_datetime"] = pd.to_datetime(df['start_datetime'])
    df["hour"] = df['start_datetime'].dt.hour
    df["day"] = df['start_datetime'].dt.day
    print(df.head())
    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')
    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(opposite_count="count", opposite_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    df_call = df[df["calltype_id"] == 1].copy()
    tmp = df_call.groupby("phone_no_m")["imei_m"].agg(voccalltype1="count", imeis="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    phone_no_m["voc_calltype1"] = phone_no_m["voccalltype1"] / phone_no_m["opposite_count"]
    tmp = df_call.groupby("phone_no_m")["city_name"].agg(city_name_call="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df_call.groupby("phone_no_m")["county_name"].agg(county_name_call="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")


    tmp = df.groupby(["phone_no_m", "opposite_no_m"])["call_dur"].agg(count="count", sum="sum")
    phone2opposite = tmp.groupby("phone_no_m")["count"].agg(phone2opposite_mean="mean", phone2opposite_median="median",
                                                            phone2opposite_max="max")
    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")
    phone2opposite = tmp.groupby("phone_no_m")["sum"].agg(phone2oppo_sum_mean="mean", phone2oppo_sum_median="median",
                                                          phone2oppo_sum_max="max")
    phone_no_m = phone_no_m.merge(phone2opposite, on="phone_no_m", how="left")


    tmp = df.groupby("phone_no_m")["call_dur"].agg(call_dur_mean="mean", call_dur_median="median", call_dur_max="max",
                                                   call_dur_min="min")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["city_name"].agg(city_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["county_name"].agg(county_name_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df.groupby("phone_no_m")["calltype_id"].agg(calltype_id_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")


    tmp = df.groupby("phone_no_m")["hour"].agg(voc_hour_mode=lambda x: stats.mode(x)[0][0],
                                               voc_hour_mode_count=lambda x: stats.mode(x)[1][0],
                                               voc_hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["day"].agg(voc_day_mode=lambda x: stats.mode(x)[0][0],
                                              voc_day_mode_count=lambda x: stats.mode(x)[1][0],
                                              voc_day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    return phone_no_m


def get_sms_feats(df):

    print(df.head())
    df['request_datetime'] = pd.to_datetime(df['request_datetime'])
    df["hour"] = df['request_datetime'].dt.hour
    df["day"] = df['request_datetime'].dt.day


    phone_no_m = df[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')

    tmp = df.groupby("phone_no_m")["opposite_no_m"].agg(sms_count="count", sms_nunique="nunique")
    tmp["sms_rate"] = tmp["sms_count"] / tmp["sms_nunique"]
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    calltype2 = df[df["calltype_id"] == 2].copy()
    calltype2 = calltype2.groupby("phone_no_m")["calltype_id"].agg(calltype_2="count")
    phone_no_m = phone_no_m.merge(calltype2, on="phone_no_m", how="left")
    phone_no_m["calltype_rate"] = phone_no_m["calltype_2"] / phone_no_m["sms_count"]

    tmp = df.groupby("phone_no_m")["hour"].agg(hour_mode=lambda x: stats.mode(x)[0][0],
                                               hour_mode_count=lambda x: stats.mode(x)[1][0],
                                               hour_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    tmp = df.groupby("phone_no_m")["day"].agg(day_mode=lambda x: stats.mode(x)[0][0],
                                              day_mode_count=lambda x: stats.mode(x)[1][0],
                                              day_nunique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    return phone_no_m


def get_user_feats(df):
    print(df.head())
    phones_app = df[["phone_no_m"]].copy()
    phones_app = phones_app.drop_duplicates(subset=['phone_no_m'], keep='last')

    phones_app['arpu_mean'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].mean(axis=1)

    phones_app['arpu_var'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].var(axis=1)
    phones_app['arpu_max'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].max(axis=1)
    phones_app['arpu_min'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].min(axis=1)
    phones_app['arpu_median'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].median(axis=1)
    phones_app['arpu_sum'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].sum(axis=1)
    phones_app['arpu_skew'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].skew(axis=1)

    phones_app['arpu_sem'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].sem(axis=1)
    phones_app['arpu_quantile'] = df[['arpu_201908', 'arpu_201909','arpu_201910','arpu_201911',
                          'arpu_201912','arpu_202001','arpu_202002','arpu_202003']].quantile(axis=1)


    return phones_app


def feats():
    train_voc = pd.read_csv(path + 'train/train_voc.csv', )
    train_voc_feat = get_voc_feat(train_voc)
    train_voc_feat.to_csv(user_data + "train_voc_feat.csv", index=False)

    train_app = pd.read_csv(path + 'train/train_app.csv', )
    train_app_feat = get_app_feats(train_app)
    train_app_feat.to_csv(user_data + "train_app_feat.csv", index=False)

    train_sms = pd.read_csv(path + 'train/train_sms.csv', )
    train_sms_feat = get_sms_feats(train_sms)
    train_sms_feat.to_csv(user_data + "train_sms_feat.csv", index=False)

    train_user = pd.read_csv(path + 'train/train_user.csv', )
    train_user_feat = get_user_feats(train_user)
    train_user_feat.to_csv(user_data + "train_user_feat.csv", index=False)
    print('feat extraction succeed!')


def merge_feat(path_feat,df):
    df_feat=pd.DataFrame(pd.read_csv(path_feat))
    return df.merge(df_feat,on='phone_no_m',how='left')


def feat_merge():
    df_user = pd.DataFrame(pd.read_csv(path + 'train/train_user.csv'))[["phone_no_m"]].copy()

    new_user = merge_feat(path+'user_data/train_voc_feat.csv', df_user)

    new_user = merge_feat(path+'user_data/train_sms_feat.csv', new_user)

    new_user = merge_feat(path+'user_data/train_app_feat.csv', new_user)

    new_user = merge_feat(path+'user_data/train_user_feat.csv', new_user)

    train_user = pd.DataFrame(pd.read_csv(path+'train/train_user.csv'))
    new_user = new_user.merge(train_user.loc[:,['phone_no_m','label']],on='phone_no_m',how='left')

    new_user.to_csv(user_data + "all_feat_with_label.csv", index=False)



def feat_normalize(df,feat='feat'):
    feat_diff_list = norm.cdf(df[feat], loc=df[feat].mean(),scale=df[feat].std())
    feat_diff_list = pd.DataFrame(feat_diff_list, columns=[feat+'_normalize'])
    feat_diff_list['phone_no_m'] = df['phone_no_m'].copy()
    return feat_diff_list


def feat_agg(df,**kwargs):
    feature_normalize = pd.DataFrame(columns=['phone_no_m'])
    feature_normalize['phone_no_m'] = df['phone_no_m'].copy()
    feature_normalize['all_normalize']=1
    for k,v in kwargs.items():
        f1_normalize=feat_normalize(df,v)
        feature_normalize = feature_normalize.merge(f1_normalize, on="phone_no_m", how="left")
        feature_normalize['all_normalize']*= feature_normalize[v+'_normalize']

    feature_normalize['all_normalize'] = pd.DataFrame(
        norm.cdf(feature_normalize['all_normalize'], loc=feature_normalize['all_normalize'].mean(),
                 scale=feature_normalize['all_normalize'].std()))

    return feature_normalize


def feat_aggregation(df,selected_feature):
    feature_normalize = pd.DataFrame(columns=['phone_no_m'])
    feature_normalize['phone_no_m'] = df['phone_no_m'].copy()
    feature_normalize['all_normalize']=1
    for v in selected_feature:
        f1_normalize=feat_normalize(df,v)
        feature_normalize = feature_normalize.merge(f1_normalize, on="phone_no_m", how="left")
        feature_normalize['all_normalize']*= feature_normalize[v+'_normalize']


    feature_normalize['all_normalize'] = pd.DataFrame(
        norm.cdf(feature_normalize['all_normalize'], loc=feature_normalize['all_normalize'].mean(),
                 scale=feature_normalize['all_normalize'].std()))
    return feature_normalize


def compute_squared_EDM_method4(X):
  X=X.T
  m,n = X.shape
  G = np.dot(X.T, X)
  H = np.tile(np.diag(G), (n,1))
  return np.sqrt(H + H.T - 2*G)


def feature_to_adj(df,args):
    '''scheme1：f1*f2*f3....'''
    if args.scheme==1:
        all_feature=df.loc[:,'all_normalize'].tolist()
        matrix_1=np.tile(all_feature,(len(all_feature),1))
        adj=abs(matrix_1 - matrix_1.T)
        threshold=args.theta

    # scheme 2：L2 distance
    elif args.scheme==2:
        data={}
        for i in range(1, len(df.columns.tolist()) - 1):
            data['f' + str(i)] = df.iloc[:, i + 1]
        df_feature_for_L2 = pd.DataFrame(data=data)

        L2_distance_matrix=compute_squared_EDM_method4(df_feature_for_L2)
        adj=L2_distance_matrix
        # threshold=0.2
        threshold=args.theta
    else:
        raise Exception("Invalid scheme!")

    adj = np.where(adj < threshold, 1, 0)

    row, col = np.diag_indices_from(adj)
    adj[row, col] = 0

    adj=np.triu(adj, 1)

    # save as sparse matrix
    allmatrix_sp = sparse.csr_matrix(adj)
    sparse.save_npz(r'./data/user_data/node_adj_sparse.npz', allmatrix_sp)
    G=nx.from_numpy_matrix(adj)

    print(nx.info(G))
    # plot graph
    # nx.draw(G, with_labels=False, font_weight='bold', node_size=1, node_color='b', edge_color='r')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_feature', type=int, default=8,
                        help='number of features to construct graph')
    parser.add_argument('--theta', type=float, default=0.2,
                        help='threshold')
    parser.add_argument('--scheme', type=int, default=2,
                        help='scheme,1 for scheme1, 2 for scheme2')
    args = parser.parse_args()

    # extract features
    feats()
    # read the features and merge them, add labels,save as csv
    feat_merge()

    all_feat = pd.DataFrame(pd.read_csv(path + 'user_data/all_feat_with_label.csv'))

    feature_index=all_feat.columns.tolist()[1:56]
    # Construct a dictionary of building graph features based on number_feature
    build_graph_feature=['voc_calltype1', 'phone2opposite_mean', 'phone2opposite_max',
                         'voc_hour_nunique', 'voc_day_nunique', 'hour_nunique',
                         'day_nunique', 'month_ids', 'opposite_unique',
                         'imeis', 'city_name_call', 'county_name_call',
                         'phone2oppo_sum_mean', 'phone2oppo_sum_median',
                         'phone2oppo_sum_max', 'city_name_nunique', 'opposite_count',
                         'voccalltype1', 'phone2opposite_median']

    if args.scheme==2:
        selected_feature=build_graph_feature[0:args.number_feature]
    elif args.scheme==1:
        selected_feature=['opposite_unique','day_nunique']
    else:
        raise Exception("Invalid scheme!")
    df=feat_aggregation(all_feat,selected_feature)

    feature_to_adj(df,args)