from hmmlearn.hmm import GaussianHMM
from pylab import *
import time
import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import xlwt
import pandas as pd
import seaborn as sns
import warnings
sns.set_style('white')
warnings.filterwarnings('ignore')

# 获取数据
data = pd.read_excel('TF DATA HMM.xlsx')
dataRATE = pd.read_excel('RATE HMM.xls')
dcol = data.columns
# 日期列表
TradeDate = pd.to_datetime(data['日期'])
tradeDate = pd.to_datetime(dataRATE['日期'])
# 总天数
days = len(TradeDate)
daysR = len(tradeDate)
days_diff = days-daysR
TradeDate = TradeDate[days_diff:]
days = daysR
contract = ['']*days
# 合同数
num_con = len(dcol)//7

# 筛选主力合约
for i in range(days):
    maxshares = 0
    for j in range(num_con):
        code_j = 1300+100*((j+3)//4)
        if j % 4 == 0:
            code_j += 12
        else:
            code_j += 3*(j % 4)
        if data[dcol[3+7*j]][i+days_diff] > maxshares:
            maxshares = data[dcol[3+7*j]][i+days_diff]
            contract[i] = 'TF%s' % code_j

# 分别获取开盘价，收盘价，持仓量，成交量
openIndex = np.zeros(days)
closeIndex = np.zeros(days)
holdShares = np.zeros(days)
dealVolume = np.zeros(days)
highest = np.zeros(days)
lowest = np.zeros(days)
for i in range(days):
    openIndex[i] = data['%s.开盘价' % contract[i]][i]
    closeIndex[i] = data['%s.收盘价' % contract[i]][i]
    holdShares[i] = data['%s.持仓量' % contract[i]][i]
    dealVolume[i] = data['%s.成交量' % contract[i]][i]
    highest[i] = data['%s.最高价' % contract[i]][i]
    lowest[i] = data['%s.最低价' % contract[i]][i]

# 对以上四个量分别取一日和五日对数差
logOpen1 = np.array(np.diff(np.log(openIndex)))
logOpen1 = logOpen1[4:]
logOpen5 = np.log(np.array(openIndex[5:])) - np.log(np.array(openIndex[:-5]))
logReturn1 = np.array(np.diff(np.log(closeIndex)))
logReturn1 = logReturn1[4:]
logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))
logShares1 = np.array(np.diff(np.log(holdShares)))
logShares1 = logShares1[4:]
logShares5 = np.log(np.array(holdShares[5:]))-np.log(np.array(holdShares[:-5]))
logVolume1 = np.array(np.diff(np.log(dealVolume)))
logVolume1 = logVolume1[4:]
logVolume5 = np.log(np.array(dealVolume[5:])) - np.log(np.array(dealVolume[:-5]))
diffMost = highest - lowest
diffMost = diffMost[5:]

# 由于五日差的存在，数据统一从第五天后开始算，总天数减少5天
days = days-5
Date = TradeDate[5:]
con = contract[5:]
openi = openIndex[5:]
close = closeIndex[5:]
hold = holdShares[5:]
volume = dealVolume[5:]
high = highest[5:]
low = lowest[5:]

# 获取增加的变量，并从第五天开始算起
Due1year = dataRATE['中债国债到期收益率:1年'][5:]
Due10years = dataRATE['中债国债到期收益率:10年'][5:]
Due5years = dataRATE['中债国债到期收益率:5年'][5:]
FR007 = dataRATE['回购定盘利率:7天(FR007)'][5:]
Due3months = dataRATE['中债国债到期收益率:3个月'][5:]
Ini10years = dataRATE['中债国开债到期收益率:10年'][5:]
LPR = dataRATE['贷款基础利率(LPR):1年'][5:]
diff_10years1year = dataRATE['中债国债到期收益率:10年:-中债国债到期收益率:1年'][5:]
diff_10years5years = dataRATE['中债国债到期收益率:10年:-中债国债到期收益率:5年'][5:]
diff_RATE = dataRATE['贷款基础利率(LPR):1年:-中债国开债到期收益率:10年'][5:]
HRB = dataRATE['价格:螺纹钢:HRB400 20mm:全国'][5:]
HN_IPE = dataRATE['南华工业品指数'][5:]
coaltot = dataRATE['日均耗煤量:6大发电集团:合计'][5:]

# 对增加的变量再做处理
diff_5years1year = Due5years-Due1year
ratio_1year3months = Due1year/Due3months
ratio_FR10years = FR007/Due10years
ratio_FR5years = FR007/Due5years
ratio_FR1year = FR007/Due1year

# 获取交易期间的总天数DAYS
startday = '%s' % Date[5+days_diff]
startday = startday[:10]
endday = '%s' % Date[days+days_diff+4]
endday = endday[:10]
startday_ = datetime.datetime.strptime(startday, '%Y-%m-%d')
endday_ = datetime.datetime.strptime(endday, '%Y-%m-%d')
interval_ = endday_ - startday_
DAYS = interval_.days

# 记录下变量
res = pd.DataFrame({'TradeDate': Date, 'contract': con}).set_index('TradeDate')
res['openIndex'] = openi
res['logOpen'] = logOpen1
res['closeIndex'] = close
res['logReturn'] = logReturn1
res['holdShares'] = hold
res['logShares'] = logShares1
res['Volumes'] = volume
res['logVolume'] = logVolume1
res['diffMost'] = diffMost

X = np.column_stack([openi, logOpen1, logOpen5, close, logReturn1, logReturn5,
                     hold, logShares1, logShares5, volume, logVolume1, logVolume5, diffMost,
                     Due1year, Due10years, Due5years, FR007, Due3months, Ini10years, LPR,
                     diff_10years1year, diff_10years5years, diff_RATE, diff_5years1year,
                     ratio_1year3months, ratio_FR10years, ratio_FR5years, ratio_1year3months,
                     HRB, HN_IPE, coaltot])

# 循环模型10次，从10次的数据中得到最好的一组数据输出，以净值、胜率、最大回撤、夏普率为判断参数，超过两个及以上个参数更优，则输出策略
count = 0
netmax = 0.
winsmax = 0.
drawbackmin = 1.
sharpemax = -10000.
annualmax = 0.
# 假设无风险投资利略为2.5%
riskfree = 0.025
retratio = np.zeros(days-1)

for k in range(100):
    # 根据不同变量的数据建模，设隐含状态有16个
    model = GaussianHMM(n_components=16, covariance_type="spherical", n_iter=10000000).fit(X)

    # 求出隐含状态
    hidden_states = model.predict(X)
    res['hidden_states'] = hidden_states

    # 按最后一天的累积利率排序，得出每个隐含状态的利率曲线
    retcumsum = np.arange(model.n_components, dtype=float)

    tf = res.closeIndex - res.openIndex
    for i in range(model.n_components):
        idx = (hidden_states == i)
        idx = np.append(0, idx[:-1])
        res['sig_ret%s' % i] = tf.multiply(idx, axis=0)
        retcumsum[i] = res['sig_ret%s' % i].cumsum()[-1]

    # 根据最后一天的累积利率的大小排序
    seq = np.arange(model.n_components)
    for i in range(model.n_components):
        for j in range(0, i):
            if retcumsum[i] > retcumsum[j]:
                seq[i], seq[j] = seq[j], seq[i]
                retcumsum[i], retcumsum[j] = retcumsum[j], retcumsum[i]

    for i in range(model.n_components):
        idx = (hidden_states == seq[i])
        idx = np.append(0, idx[:-1])
        res['sig_ret%s' % i] = tf.multiply(idx, axis=0)

    # 下面制定策略
    judge = np.zeros(model.n_components)
    for i in range(model.n_components):
        # 根据每个隐含状态的累积利率在0以上或以下的天数判断做多或做空策略
        longdays = 0
        shortdays = 0
        idx = (hidden_states == seq[i])
        for j in range(days):
            if idx[j] == 1 and res['sig_ret%s' % i].cumsum()[j] > 0:
                longdays += 1
            if idx[j] == 1 and res['sig_ret%s' % i].cumsum()[j] < 0:
                shortdays += 1
        if longdays > shortdays+0.05*days or longdays > 1.2*shortdays:
            judge[i] = 1
        if shortdays > longdays+0.05*days or shortdays > 1.2*longdays:
            judge[i] = -1
    idxlong = np.zeros(days)
    idxshort = np.zeros(days)
    # judge=1做多，judge=-1做空
    for i in range(model.n_components):
        if judge[i] > 0:
            idxlong += (hidden_states == seq[i])
        if judge[i] < 0:
            idxshort += (hidden_states == seq[i])
    # 获得状态结果后第二天进行买入操作
    res['long_sig'] = (idxlong > 0)
    idxlong = np.append(0, idxlong[:-1])
    # 获得状态结果后第二天进行买入操作
    res['short_sig'] = (idxshort > 0)
    idxshort = np.append(0, idxshort[:-1])

    res['today_diff'] = tf
    res['sig_retbest'] = tf.multiply(idxlong, axis=0) - tf.multiply(idxshort, axis=0)
    res['pre_interest'] = res['sig_retbest'].cumsum()
    tf[0] = 0
    res['real_interest'] = tf.cumsum()

    # 本金1,000,000，每天以10%的仓位开仓，收盘平仓
    diss = 0
    long_sig = res['long_sig']
    short_sig = res['short_sig']
    ini_money = 1000000
    store = 0.1
    money = np.zeros(days)
    money[0] = ini_money
    shares = np.zeros(days)
    shares[0] = 0
    win = 0
    drawback = 0
    for i in range(days - 1):
        # 每天交易的手数
        shares[i + 1] = (money[i] * store) // (200 * close[i])
        money[i + 1] = money[i]
        stop = 0
        # 看涨
        if long_sig[i] and not short_sig[i]:
            trademoney = 10000 * shares[i + 1] * openi[i + 1]
            money[i + 1] = money[i] - trademoney
            if high[i + 1] >= 1.008 * openi[i + 1]:
                trademoney = 10000 * shares[i + 1] * 1.008 * openi[i + 1]
            elif low[i + 1] <= 0.995 * openi[i + 1]:
                trademoney = 10000 * shares[i + 1] * 0.995 * openi[i + 1]
            else:
                trademoney = 10000 * shares[i + 1] * close[i + 1]
            money[i + 1] += trademoney
            money[i + 1] -= 10 * shares[i + 1]
        # 看跌
        if not long_sig[i] and short_sig[i]:
            trademoney = 10000 * shares[i + 1] * openi[i + 1]
            money[i + 1] = money[i] + trademoney
            if low[i + 1] <= 0.992 * openi[i + 1]:
                trademoney = 10000 * shares[i + 1] * 0.992 * openi[i + 1]
            elif high[i + 1] >= 1.005 * openi[i + 1]:
                trademoney = 10000 * shares[i + 1] * 1.005 * openi[i + 1]
            else:
                trademoney = 10000 * shares[i + 1] * close[i + 1]
            money[i + 1] -= trademoney
            money[i + 1] -= 10 * shares[i + 1]
        if money[i + 1] > money[i]:
            win += 1
        else:
            # 如果这一天的策略失败，计算这一天为止的最大回撤
            todaydrawback = (money[i] - money[i + 1]) / money[i]
            if todaydrawback > drawback:
                drawback = todaydrawback

    res['money'] = np.array(money)
    res['shares'] = np.array(shares)

    netvalue = money[-1] / ini_money
    rateofwin = win / (days - 1)
    loses = days - 1 - win
    annualizedreturn = ((netvalue - 1) / DAYS) * 365
    ret = np.diff(money)
    for i in range(days - 1):
        retratio[i] = ret[i] / money[i]
    res['retratio'] = np.append(0, retratio)
    volatility_yr = np.std(retratio, ddof=0) * np.sqrt(252)
    sharpe = (annualizedreturn - riskfree) / volatility_yr

    print('净值：%.2f' % netvalue, end='   ')
    print('此时最大净值：%.2f' % netmax)
    print('年化收益率：%.2f%%' % (100 * annualizedreturn), end='   ')
    print('此时最大年化收益率：%.2f%%' % (100 * annualmax))
    print('胜率：%.2f%%' % (100 * rateofwin), end='   ')
    print('此时最大胜率：%.2f%%' % (100 * winsmax))
    print('最大回撤：%.2f%%' % (100 * drawback), end='   ')
    print('此时最大回撤：%.2f%%' % (100 * drawbackmin))
    print('夏普率：%.2f' % sharpe, end='   ')
    print('此时最大夏普率：%.2f' % sharpemax)
    print('盈利次数：%d' % win)
    print('亏损次数：%d' % loses)

    if netvalue >= netmax:
        diss += 1
    if rateofwin >= winsmax:
        diss += 1
    if drawback <= drawbackmin:
        diss += 1
    if sharpe >= sharpemax:
        diss += 1

    if (diss > 2) and (rateofwin > 0.5):
        count += 1
        netmax = netvalue
        winsmax = rateofwin
        drawbackmin = drawback
        sharpemax = sharpe
        annualmax = annualizedreturn
        count += 1
        print('第%d次循环，找到更好的模型' % (k + 1))

        figure(3 * count - 2)
        for i in range(model.n_components):
            plt.plot(res['sig_ret%s' % i].cumsum(), label='%dth hidden state' % i)
            plt.legend()
            plt.grid(1)
        plt.savefig('Hidden_States.eps')

        figure(3 * count - 1)
        plt.plot_date(Date, res['real_interest'], '-', label='accumulated return')
        plt.plot_date(Date, res['pre_interest'], '-', label='our return')
        plt.legend()
        plt.grid(1)
        plt.savefig('Return.eps')

        figure(3 * count)
        plt.plot_date(Date, money, '-', label='Strategy')
        plt.legend()
        plt.grid(1)
        plt.savefig('Strategy.eps')

        res.to_excel('Strategy.xls')
    else:
        print('第%d次循环，不是更好的模型' % (k + 1))

print('---------------------------------------------------------------------------------------------------------------')
stat = pd.DataFrame({'开始时间': [startday], '结束时间': [endday], '天数': [DAYS], '交易日数': [days],
                     '净值': ['%.2f' % netmax], '胜率': ['%.2f%%' % (100 * winsmax)],
                     '最大回撤': ['%.2f%%' % (100 * drawbackmin)], '年化收益率': ['%.2f%%' % (100*annualmax)],
                     '夏普率': ['%.2f' % sharpemax]})

print('最终的结果：')
print('净值：%.2f' % netmax)
print('年化收益率：%.2f%%' % (100*annualmax))
print('胜率：%.2f%%' % (100 * winsmax))
print('最大回撤：%.2f%%' % (100 * drawbackmin))
print('夏普率：%.2f' % sharpemax)

stat.to_excel('parameter.xls')
