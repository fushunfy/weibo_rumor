# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import jieba
import pickle
import lightgbm as lgb
from catboost import CatBoostClassifier
from keras_bert import load_trained_model_from_checkpoint
import tokenization  # Actually keras_bert contains tokenization part, here just for convenience
import os
from keras.callbacks import Callback
import keras
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import tensorflow as tf

test_name = 'data/test.csv'
test = pd.read_csv(test_name)

test['userLocation2'] = test['userLocation'].fillna("nan").map(lambda x: " " in x)
test['userLocation3'] = test['userLocation'].fillna("nan").map(lambda x: " " not in x and (len(x) > 3))

test['All_text'] = test['text'] + " * " + test['userDescription']
test['All_text'].fillna('-1', inplace=True)

test['all_text_len_num'] = test['All_text'].apply(lambda x: len(x))
test['userDescription_len_num'] = test['userDescription'].apply(lambda x: len(x) if x is not np.nan else 0)


def cut_words(x):
    # 用空格分词
    return ' '.join(jieba.cut(x))


n_components = [100, 20, 50]
text_features = []

text_feas = ['text', 'userDescription', 'All_text']

X_text = test[text_feas]

for a, i in enumerate(text_feas):
    # Initialize decomposition methods:
    X_text[i] = X_text[i].astype(str)
    X_text[i] = X_text[i].apply(lambda x: cut_words(x))

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
text_features = pd.concat(text_features, axis=1)

test = pd.concat([test, text_features], axis=1)

drop_feas = ['id', 'label', 'piclist']
features = [i for i in test.columns if i not in drop_feas]
print(features)

object_feas = [i for i in features if str(test[i].dtype) == 'object']

print(object_feas)

for fea in object_feas:
    test[fea] = pd.factorize(test[fea])[0]
    test[fea + "_freq"] = test[fea].map(test[fea].value_counts(dropna=False))

features = [i for i in test.columns if i not in drop_feas]

print(features)

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
print(test['label'].value_counts())

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
print(lgb_pred_result['acc'].value_counts())
lgb_acc_num = lgb_pred_result['acc'].value_counts()
lgb_accuracy = 1 - lgb_acc_num[0] / (lgb_acc_num[0] + lgb_acc_num[1])
print('accuracy: {:.2%}'.format(lgb_accuracy))

if os.path.exists('result/'):
    os.removedirs(r"result/")
os.makedirs(r"result/")

with open('result/lgb_result.txt', 'w') as file_obj:
    file_obj.write('lgb_accuracy: {:.2%}'.format(lgb_accuracy))

dev_data = pd.read_csv(test_name)
cat_pred_result = pd.merge(dev_data, cat_result.loc[:, ['id', 'cat_label']], how='left', on='id')
cat_pred_result['acc'] = cat_pred_result.apply(lambda x: 1 if x['label'] == x['cat_label'] else 0, axis=1)
print(cat_pred_result['acc'].value_counts())
cat_acc_num = cat_pred_result['acc'].value_counts()
cat_accuracy = 1 - cat_acc_num[0] / (cat_acc_num[0] + cat_acc_num[1])
print('accuracy: {:.2%}'.format(cat_accuracy))
with open('result/cat_result.txt', 'w') as file_obj:
    file_obj.write('cat_accuracy: {:.2%}'.format(cat_accuracy))

dev_data = pd.read_csv(test_name)
lgb_cat_pred_result = pd.merge(dev_data, lgb_cat_result.loc[:, ['id', 'lgb_cat_label']], how='left', on='id')
lgb_cat_pred_result['acc'] = lgb_cat_pred_result.apply(lambda x: 1 if x['label'] == x['lgb_cat_label'] else 0, axis=1)
print(lgb_cat_pred_result['acc'].value_counts())
lgb_cat_acc_num = lgb_cat_pred_result['acc'].value_counts()
lgb_cat_accuracy = 1 - lgb_cat_acc_num[0] / (lgb_cat_acc_num[0] + lgb_cat_acc_num[1])
print('accuracy: {:.2%}'.format(lgb_cat_accuracy))
with open('result/lgb_cat_result.txt', 'w') as file_obj:
    file_obj.write('lgb_cat_accuracy: {:.2%}'.format(lgb_cat_accuracy))

tmp = test[test['pred'] >= 0.9]
print(len(tmp))
tmp1 = test[test['pred'] <= 0.1]
print(len(tmp1))

add_data_test = pd.concat([tmp, tmp1])

add_data_test['label'] = (add_data_test['pred'] > 0.5).astype(int)

add_data_test['label'].value_counts()

del add_data_test['pred']

if os.path.exists('data/pesudo_data.csv'):
    os.remove("data/pesudo_data.csv")

add_data_test.to_csv("data/pesudo_data.csv", index=False)

BERT_PRETRAINED_DIR = './chinese_L-12_H-768_A-12/'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

maxlen = 200

config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=maxlen)  #


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


# 2分类的f1
# F1ScoreCallback(validation = (val_x, val_y)
class F1ScoreCallback(Callback):
    def __init__(self, validation, predict_batch_size=20, include_on_batch=False):
        super(F1ScoreCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

        print('validation shape', len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('avg_f1_score_val' in self.params['metrics']):
            self.params['metrics'].append('avg_f1_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['avg_f1_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['avg_f1_score_val'] = float('-inf')

        if (self.validation):
            y_predict = self.model.predict(self.validation[0])
            y_predict = (y_predict > 0.5).astype(int)
            y_true = (self.validation[1] > 0.5).astype(int)
            f1 = f1_score(y_true, y_predict)
            # print("macro f1_score %.4f " % f1)
            #             f2 = f1_score(self.validation[1], y_predict, average='micro')
            # print("micro f1_score %.4f " % f2)
            avgf1 = f1
            # print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] = avgf1

            print("current f1 %f" % avgf1)


# 自定义优化器, 在keras中实现差分学习率
class AdamWarmup(keras.optimizers.Optimizer):
    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                 amsgrad=False, **kwargs):
        super(AdamWarmup, self).__init__(**kwargs)
        # 命名域 (name scope)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
            self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_kernel_weight_decay = kernel_weight_decay
        self.initial_bias_weight_decay = bias_weight_decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        # 获取梯度
        grads = self.get_gradients(loss, params)
        # 定义赋值算子集合
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1
        # 根据一个标量值在两个操作之间切换。switch接口，顾名思义，就是一个if/else条件判断语句。不过要求输入和输出都必须是张量。
        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.lr * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
            )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))
        # zero init of 1st moment
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        # # zero init of exponentially weighted infinity norm
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if 'bias' in p.name or 'Norm' in p.name:
                if self.initial_bias_weight_decay > 0.0:
                    p_t += self.bias_weight_decay * p
            else:
                if self.initial_kernel_weight_decay > 0.0:
                    p_t += self.kernel_weight_decay * p
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'decay_steps': float(K.get_value(self.decay_steps)),
            'warmup_steps': float(K.get_value(self.warmup_steps)),
            'min_lr': float(K.get_value(self.min_lr)),
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'kernel_weight_decay': float(K.get_value(self.kernel_weight_decay)),
            'bias_weight_decay': float(K.get_value(self.bias_weight_decay)),
            'amsgrad': self.amsgrad,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
print(bert_pred_result['acc'].value_counts())
bert_acc_num = bert_pred_result['acc'].value_counts()
bert_accuracy = 1 - bert_acc_num[0] / (bert_acc_num[0] + bert_acc_num[1])
print('accuracy: {:.2%}'.format(bert_accuracy))
with open('result/bert_weights_result.txt', 'w') as file_obj:
    file_obj.write('bert_weights_accuracy: {:.2%}'.format(bert_accuracy))

# Do some code, e.g. train and save model


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)

oof_train = np.zeros((len(train), 1))
oof_test = np.zeros((len(test), 1))
# 这里kf.split(X)返回的是X中进行分裂后train和test的索引值，令X中数据集的索引为0，1，2，3；第一次分裂，先选择test，
# 索引为0和1的数据集为test,剩下索引为2和3的数据集为train；第二次分裂，先选择test，索引为2和3的数据集为test,剩下索引为0和1的数据集为train
for k, (idx_train, idx_test) in enumerate(skf.split(train, train['label'])):
    K.clear_session()
    tf.reset_default_graph()
    # iloc函数：通过行号来取行数据
    tr_df = train.iloc[idx_train]
    te_df = train.iloc[idx_test]

    train_lines, train_labels = tr_df['text'].values, tr_df['label'].astype(int).values

    print('sample used', train_lines.shape)
    token_input, mask_input = convert_lines(train_lines, maxlen, tokenizer)
    seg_input = np.zeros((token_input.shape[0], maxlen))
    # mask_input = np.ones((token_input.shape[0],maxlen))
    print(token_input.shape)
    print(seg_input.shape)
    print(mask_input.shape)
    print('begin training')

    train_x = [token_input, seg_input]
    # train_y=to_categorical(train_labels).astype(int)
    train_y = train_labels

    print(train_y.shape)

    val_lines, val_labels = te_df['text'].values, te_df['label'].astype(int).values

    print('val sample used', val_lines.shape)
    val_token_input, val_mask_input = convert_lines(val_lines, maxlen, tokenizer)
    val_seg_input = np.zeros((val_token_input.shape[0], maxlen))
    # mask_input = np.ones((token_input.shape[0], maxlen))
    print(val_token_input.shape)
    print(val_seg_input.shape)
    print(val_mask_input.shape)
    print('begin training')

    val_x = [val_token_input, val_seg_input]
    # val_y=to_categorical(val_labels).astype(int)
    val_y = val_labels

    print(val_y.shape)

    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    # 将 BERT 模型载入到 keras
    bert_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=maxlen)  #
    # model.summary(line_length=120)
    for l in bert_model.layers:
        l.trainable = True
    # Input() 用于实例化 Keras 张量。
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    # 将值输入bert模型
    x = bert_model([x1_in, x2_in])
    # # 取出[CLS]对应的向量用来做分类, 将任意表达式封装为 Layer 对象。对上一层的输出施以任何Theano/TensorFlow函数
    # 各种基本层，除了可以add进Sequential容器串联之外，它们本身也是callable对象，被调用之后，返回的还是callable对象。
    # 所以可以将它们视为函数，通过调用的方式来进行串联。
    # x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据,
    x = Lambda(lambda x: x[:, 0])(x)
    # 该层有一个神经元， activation: 该层使用的激活函数, Dense层： 全连接层
    # nD 张量，尺寸: (batch_size, ..., input_dim)。 最常见的情况是一个尺寸为 (batch_size, input_dim) 的 2D 输入。
    p = Dense(1, activation='sigmoid')(x)

    model3 = Model([x1_in, x2_in], p)

    # 用足够小的学习率
    lr = (1e-5)
    weight_decay = 0.001
    nb_epochs = 3
    bsz = 23
    decay_steps = int(nb_epochs * train_lines.shape[0] / bsz)
    warmup_steps = int(0.1 * decay_steps)

    # 自定义优化器
    adamwarm = AdamWarmup(lr=lr, decay_steps=decay_steps, warmup_steps=warmup_steps, kernel_weight_decay=weight_decay)
    """ ModelCheckpoint: 保存训练过程中的最佳模型权重
        monitor：需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
        verbose：信息展示模式，0或1。为1表示输出epoch模型保存信息，默认为0表示不输出该信息，信息形如：
        Epoch 00001: val_acc improved from -inf to 0.49240, saving model to /xxx/checkpoint/model_001-0.3902.h5
        save_best_only: 当设置为True时，将只保存在验证集上性能最好的模型
        mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，
              当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
        save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
         """
    model3.load_weights('model/bert_weights/best_model_task1_%d.h5' % (k + 1))
    # 损失函数采用binary_crossentropy，交叉熵损失函数，一般用于二分类：
    # metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]如果要在多输出模型中为不同的输出指定不同的指标，
    # 可像该参数传递一个字典，例如metrics={‘ouput_a’: ‘accuracy’}
    # 编译模型
    model3.compile(loss='binary_crossentropy', optimizer=adamwarm, metrics=['acc'])
    # 使用EarlyStopping防止过拟合
    """ monitor：需要监视的量，通常为：val_acc 或 val_loss 或 acc 或 loss
        patience：当early stop被激活（如发现loss相比上patience个epoch训练没有下降），则经过patience个epoch后停止训练。
        verbose：信息展示模式
        mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练
    """
    early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max', patience=5, verbose=1)
    # 在这里进行了shuffle操作
    """ fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
        validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试
                          的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身
                          是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
        validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
        shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，
                 它将在batch内部将数据打乱。
        batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
        epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
        callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
        
  
    """
    model3.fit(train_x, train_y, validation_data=(val_x, val_y),
               callbacks=[F1ScoreCallback(validation=(val_x, val_y)), early_stopping], batch_size=bsz, epochs=nb_epochs,
               shuffle=True)

    # 当使用predict()方法进行预测时，返回值是数值，表示样本属于每一个类别的概率
    oof_train[idx_test] = model3.predict(val_x)
    oof_test += model3.predict(test_x) / skf.n_splits

test['label'] = (oof_test > 0.5).astype(int)

if os.path.exists('./data/submission.csv'):
    os.remove("./data/submission.csv")

test[['id', 'label']].to_csv("./data/submission.csv", index=False)

dev_result = pd.read_csv("./data/submission.csv")
dev_data = pd.read_csv(test_name)
dev_result['pred'] = dev_result['label']
result = pd.merge(dev_data, dev_result.loc[:, ['id', 'pred']], how='left', on='id')

print(dev_data)

print(result[:1])

result['acc'] = result.apply(lambda x: 1 if x['label'] == x['pred'] else 0, axis=1)

print(result['acc'].value_counts())

acc_num = result['acc'].value_counts()

print(acc_num[0])

accuracy = 1 - acc_num[0] / (acc_num[0] + acc_num[1])
print('accuracy: {:.2%}'.format(accuracy))

with open('result/all_weights_result.txt', 'w') as file_obj:
    file_obj.write('all_weights_accuracy: {:.2%}'.format(accuracy))
