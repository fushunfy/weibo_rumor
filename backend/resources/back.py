# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import jieba
import pickle
import lightgbm as lgb
from catboost import CatBoostClassifier
from keras_bert import load_trained_model_from_checkpoint
import tokenization
import os
from keras.layers import Dense, Input, Lambda
from keras.models import Model

test_name = 'data/test.csv'
test = pd.read_csv(test_name)

test['userLocation2'] = test['userLocation'].fillna("nan").map(lambda x: " " in x)
test['userLocation3'] = test['userLocation'].fillna("nan").map(lambda x: " " not in x and (len(x) > 3))

test['All_text'] = test['text'] + " * " + test['userDescription']
test['All_text'].fillna('-1', inplace=True)

test['all_text_len_num'] = test['All_text'].apply(lambda x: len(x))
test['userDescription_len_num'] = test['userDescription'].apply(lambda x: len(str(x)) if x is not np.nan else 0)


def cut_words(x):
    # 用空格分词
    return ' '.join(jieba.cut(x))


n_components = [100, 20, 50]
text_features = []

text_feas = ['text', 'userDescription', 'All_text']

X_text = test[text_feas]

for a, i in enumerate(text_feas):
    # Initialize decomposition methods:
    X_text.loc[:, i] = X_text.loc[:, i].astype(str)
    X_text.loc[:, i] = X_text.loc[:, i].apply(lambda x: cut_words(x))

    print('generating features from: {}'.format(i))

    with open('model/tokenize/tokenize_%s.pkl' % i, 'rb') as f:  # Python 3: open(..., 'rb')
        try:
            tfidf, svd, nmf, lda = pickle.load(f)
        except EOFError:
            print("load file error")

    tfidf_col = tfidf.transform(X_text.loc[:, i].values)
    print("SVD")
    svd_col = svd.transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
    print("NMF")
    nmf_col = nmf.transform(tfidf_col)
    nmf_col = pd.DataFrame(nmf_col)
    nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
    print("LDA")
    lda_col = lda.transform(tfidf_col)
    lda_col = pd.DataFrame(lda_col)
    lda_col = lda_col.add_prefix('LDA_{}_'.format(i))

    text_features.append(svd_col)
    text_features.append(nmf_col)
    text_features.append(lda_col)

# Combine all extracted features:
# 0是行，1是列
text_features = pd.concat(text_features, axis=1)
text_features.reset_index(drop=True, inplace=True)
test = pd.concat([test, text_features], axis=1)

drop_feas = ['id', 'label', 'piclist']
features = [i for i in test.columns if i not in drop_feas]

object_feas = [i for i in features if str(test[i].dtype) == 'object']


for fea in object_feas:
    test[fea] = pd.factorize(test[fea])[0]
    test[fea + "_freq"] = test[fea].map(test[fea].value_counts(dropna=False))

features = [i for i in test.columns if i not in drop_feas]


n_splits = 5

lgb_oof_test = np.zeros((test.shape[0], 1))
for i in range(n_splits):
    model = lgb.Booster(model_file='model/lightgbm/lightgbm_model_%d.h5' % (i + 1))
    test_pred = model.predict(test[features], num_iteration=model.best_iteration)
    lgb_oof_test += test_pred.reshape(-1, 1) / n_splits

categorical_features = object_feas
# num_features = [i for i in features if i not in categorical_features]
cat_oof_test = np.zeros((len(test), 1))
for i in range(n_splits):
    clf = CatBoostClassifier()
    clf.load_model('model/catboost/catboost_model_%d.h5' % (i + 1))
    cat_oof_test += clf.predict_proba(test[features])[:, 1].reshape(-1, 1) / n_splits

# blend lgb and catboost
test = pd.read_csv(test_name)

test['label'] = lgb_oof_test * 0.5 + cat_oof_test * 0.5
test['label'] = (test['label'] > 0.4).astype(int)

test['pred'] = lgb_oof_test * 0.5 + cat_oof_test * 0.5

test['lgb_pred'] = lgb_oof_test
test['cat_pred'] = cat_oof_test
test['lgb_cat_pred'] = test['pred']

lgb_result = pd.DataFrame({'id': test['id'], 'lgb_label': (test['lgb_pred'] > 0.5).astype(int)})
cat_result = pd.DataFrame({'id': test['id'], 'cat_label': (test['cat_pred'] > 0.5).astype(int)})
lgb_cat_result = pd.DataFrame({'id': test['id'], 'lgb_cat_label': (test['lgb_cat_pred'] > 0.5).astype(int)})

dev_data = pd.read_csv(test_name)
lgb_pred_result = pd.merge(dev_data, lgb_result.loc[:, ['id', 'lgb_label']], how='left', on='id')
lgb_pred_result['acc'] = lgb_pred_result.apply(lambda x: 1 if x['label'] == x['lgb_label'] else 0, axis=1)
lgb_acc_num = lgb_pred_result['acc'].value_counts()

lgb_acc_list = list(lgb_acc_num.items())
if len(lgb_acc_list) == 2:
    if lgb_acc_list[0][0] == 0:
        lgb_accuracy = lgb_acc_list[1][1] / (lgb_acc_list[0][1] + lgb_acc_list[1][1])
    else:
        lgb_accuracy = lgb_acc_list[0][1] / (lgb_acc_list[0][1] + lgb_acc_list[1][1])
else:
    if lgb_acc_list[0][0] == 0:
        lgb_accuracy = 0
    else:
        lgb_accuracy = 1

print('lgb accuracy: {:.2%}'.format(lgb_accuracy))
with open('result/lgb_result.txt', 'w') as file_obj:
    file_obj.write('lgb_accuracy: {:.2%}'.format(lgb_accuracy))

dev_data = pd.read_csv(test_name)
cat_pred_result = pd.merge(dev_data, cat_result.loc[:, ['id', 'cat_label']], how='left', on='id')
cat_pred_result['acc'] = cat_pred_result.apply(lambda x: 1 if x['label'] == x['cat_label'] else 0, axis=1)
cat_acc_num = cat_pred_result['acc'].value_counts()

cat_acc_list = list(cat_acc_num.items())
if len(cat_acc_list) == 2:
    if cat_acc_list[0][0] == 0:
        cat_accuracy = cat_acc_list[1][1] / (cat_acc_list[0][1] + cat_acc_list[1][1])
    else:
        cat_accuracy = cat_acc_list[0][1] / (cat_acc_list[0][1] + cat_acc_list[1][1])
else:
    if cat_acc_list[0][0] == 0:
        cat_accuracy = 0
    else:
        cat_accuracy = 1

print('cat accuracy: {:.2%}'.format(cat_accuracy))
with open('result/cat_result.txt', 'w') as file_obj:
    file_obj.write('cat_accuracy: {:.2%}'.format(cat_accuracy))

dev_data = pd.read_csv(test_name)
lgb_cat_pred_result = pd.merge(dev_data, lgb_cat_result.loc[:, ['id', 'lgb_cat_label']], how='left', on='id')
lgb_cat_pred_result['acc'] = lgb_cat_pred_result.apply(lambda x: 1 if x['label'] == x['lgb_cat_label'] else 0, axis=1)
lgb_cat_acc_num = lgb_cat_pred_result['acc'].value_counts()

lgb_cat_acc_list = list(lgb_cat_acc_num.items())
if len(lgb_cat_acc_list) == 2:
    if lgb_cat_acc_list[0][0] == 0:
        lgb_cat_accuracy = lgb_cat_acc_list[1][1] / (lgb_cat_acc_list[0][1] + lgb_cat_acc_list[1][1])
    else:
        lgb_cat_accuracy = lgb_cat_acc_list[0][1] / (lgb_cat_acc_list[0][1] + lgb_cat_acc_list[1][1])
else:
    if lgb_cat_acc_list[0][0] == 0:
        lgb_cat_accuracy = 0
    else:
        lgb_cat_accuracy = 1

print('lgb+cat accuracy: {:.2%}'.format(lgb_cat_accuracy))
with open('result/lgb_cat_result.txt', 'w') as file_obj:
    file_obj.write('lgb_cat_accuracy: {:.2%}'.format(lgb_cat_accuracy))


BERT_PRETRAINED_DIR = './chinese_L-12_H-768_A-12/'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

maxlen = 200

def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    all_mask = []
    longer = 0
    for i in range(example.shape[0]):
        tokens_a = tokenizer.tokenize(example[i])
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
            mask = [1] * (max_seq_length + 2)
        else:
            mask = [1] * (len(tokens_a) + 2) + [0] * (max_seq_length - len(tokens_a))
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (
                max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
        all_mask.append(mask)
    return np.array(all_tokens), np.array(all_mask)


row = None
nb_epochs = 1
bsz = 32
dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)

print('build tokenizer done')

add_data_test = pd.read_csv("data/pesudo_data.csv")

train = add_data_test

test = pd.read_csv(test_name)

train['text'] = train['text'].astype(str)
test['text'] = test['text'].astype(str)

train['text'] = train['text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', ' ', regex=True)
test['text'] = test['text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', ' ', regex=True)

test_lines = test['text'].values
print('sample used', test_lines.shape)
test_token_input, test_mask_input = convert_lines(test_lines, maxlen, tokenizer)
test_seg_input = np.zeros((test_token_input.shape[0], maxlen))
print(test_token_input.shape)
print(test_seg_input.shape)
print(test_mask_input.shape)
print('begin predictting')

test_x = [test_token_input, test_seg_input]

oof_test = np.zeros((len(test), 1))

config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
bert_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=maxlen)  #
# model.summary(line_length=120)
for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)

model3 = Model([x1_in, x2_in], p)

for i in range(0, 5):
    model3.load_weights('./model/bert_weights/best_model_task1_%d.h5' % (i + 1))

    oof_test += model3.predict(test_x) / 5

test['bert_label'] = oof_test
bert_result = pd.DataFrame({'id': test['id'], 'bert_label': (test['bert_label'] > 0.5).astype(int)})
dev_data = pd.read_csv(test_name)
bert_pred_result = pd.merge(dev_data, bert_result.loc[:, ['id', 'bert_label']], how='left', on='id')
bert_pred_result['acc'] = bert_pred_result.apply(lambda x: 1 if x['label'] == x['bert_label'] else 0, axis=1)
bert_acc_num = bert_pred_result['acc'].value_counts()

bert_acc_list = list(bert_acc_num.items())
if len(bert_acc_list) == 2:
    if bert_acc_list[0][0] == 0:
        bert_accuracy = bert_acc_list[1][1] / (bert_acc_list[0][1] + bert_acc_list[1][1])
    else:
        bert_accuracy = bert_acc_list[0][1] / (bert_acc_list[0][1] + bert_acc_list[1][1])
else:
    if bert_acc_list[0][0] == 0:
        bert_accuracy = 0
    else:
        bert_accuracy = 1

print('single bert accuracy: {:.2%}'.format(bert_accuracy))
with open('result/bert_weights_result.txt', 'w') as file_obj:
    file_obj.write('bert_weights_accuracy: {:.2%}'.format(bert_accuracy))

test['pred'] = oof_test * (1/3) + lgb_oof_test * (1/3) + cat_oof_test * (1/3)
test['label'] = (test['pred'] > 0.5).astype(int)

if os.path.exists('./data/submission.csv'):
    os.remove("./data/submission.csv")

test[['id', 'label']].to_csv("./data/submission.csv", index=False)

dev_result = pd.read_csv("./data/submission.csv")
dev_data = pd.read_csv(test_name)
dev_result['pred'] = dev_result['label']
result = pd.merge(dev_data, dev_result.loc[:, ['id', 'pred']], how='left', on='id')

result['acc'] = result.apply(lambda x: 1 if x['label'] == x['pred'] else 0, axis=1)

acc_num = result['acc'].value_counts()

acc_list = list(acc_num.items())
if len(acc_list) == 2:
    if acc_list[0][0] == 0:
        accuracy = acc_list[1][1] / (acc_list[0][1] + acc_list[1][1])
    else:
        accuracy = acc_list[0][1] / (acc_list[0][1] + acc_list[1][1])
else:
    if acc_list[0][0] == 0:
        accuracy = 0
    else:
        accuracy = 1

print('all model accuracy: {:.2%}'.format(accuracy))

with open('result/all_weights_result.txt', 'w') as file_obj:
    file_obj.write('all_weights_accuracy: {:.2%}'.format(accuracy))
