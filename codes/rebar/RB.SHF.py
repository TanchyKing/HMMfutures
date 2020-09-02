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
data = pd.read_excel('blackp HMM.xlsx')
dataRATE = pd.read_excel('black HMM.xls')
dcol = data.columns
# 日期列表
TradeDate = pd.to_datetime(data['日期'])
# 总天数
days = len(TradeDate)

# 第n份合约
con_n = 0
# 分别获取开盘价，收盘价，持仓量，成交量
openIndex = np.array(data[dcol[con_n * 7 + 1]])
closeIndex = np.array(data[dcol[con_n * 7 + 2]])
holdShares = np.array(data[dcol[con_n * 7 + 3]])
highest = np.array(data[dcol[con_n * 7 + 4]])
lowest = np.array(data[dcol[con_n * 7 + 5]])
dealVolume = np.array(data[dcol[con_n * 7 + 6]])

# 一年天数
year_days = 60
# 年数
years = math.ceil(days/year_days)
# 最后daysn天做交易
daysn = 60
# 从获得的数据的第dayn天开始算起
dayn = days - year_days - 5 - daysn

# 对以上四个量分别取一日和五日对数差
logOpen1 = np.array(np.diff(np.log(openIndex)))
logOpen1 = logOpen1[4:]
logOpen1 = logOpen1[dayn:]
logOpen5 = np.log(np.array(openIndex[5:])) - np.log(np.array(openIndex[:-5]))
logOpen5 = logOpen5[dayn:]
logReturn1 = np.array(np.diff(np.log(closeIndex)))
logReturn1 = logReturn1[4:]
logReturn1 = logReturn1[dayn:]
logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))
logReturn5 = logReturn5[dayn:]
logShares1 = np.array(np.diff(np.log(holdShares)))
logShares1 = logShares1[dayn:]
logShares1 = logShares1[4:]
logShares5 = np.log(np.array(holdShares[5:]))-np.log(np.array(holdShares[:-5]))
logShares5 = logShares5[dayn:]
logVolume1 = np.array(np.diff(np.log(dealVolume)))
logVolume1 = logVolume1[4:]
logVolume1 = logVolume1[dayn:]
logVolume5 = np.log(np.array(dealVolume[5:])) - np.log(np.array(dealVolume[:-5]))
logVolume5 = logVolume5[dayn:]
diffMost = highest - lowest
diffMost = diffMost[dayn+5:]

# 由于五日差的存在，数据统一从第五天后开始算，总天数减少5天
days = days-dayn-5
Date = TradeDate[dayn+5:]
openi = openIndex[dayn+5:]
close = closeIndex[dayn+5:]
hold = holdShares[dayn+5:]
volume = dealVolume[dayn+5:]
high = highest[dayn+5:]
low = lowest[dayn+5:]

res = pd.DataFrame({'TradeDate': Date}).set_index('TradeDate')
res['openIndex'] = openi
res['closeIndex'] = close
res['holdShares'] = hold
res['Volumes'] = volume
res['diffMost'] = diffMost

# 获取交易期间的总天数DAYS
startday = '%s' % Date[dayn+5]
startday = startday[:10]
endday = '%s' % Date[days+dayn+4]
endday = endday[:10]
startday_ = datetime.datetime.strptime(startday, '%Y-%m-%d')
endday_ = datetime.datetime.strptime(endday, '%Y-%m-%d')
interval_ = endday_ - startday_
DAYS = interval_.days

X = np.column_stack([openi, logOpen1, logOpen5, close, logReturn1, logReturn5,
                     hold, logShares1, logShares5, volume, logVolume1, logVolume5, diffMost])

# 获取增加的变量，并从第五天开始算起
dRcol = dataRATE.columns
ldR = len(dRcol)
for i in range(ldR - 1):
    X = np.column_stack([X, dataRATE[dRcol[i + 1]][dayn + 5:]])

res['hidden_states'] = np.zeros(days)
for i in range(16):
    res['sig_ret%s' % i] = np.zeros(days)
res['long_sig'] = np.zeros(days)
res['short_sig'] = np.zeros(days)
res['today_diff'] = np.zeros(days)
res['sig_retbest'] = np.zeros(days)
res['pre_interest'] = np.zeros(days)

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

k = 0
while (k < 5):
    # 学习第一个月的数据，做第一个月
    model = GaussianHMM(n_components=16, covariance_type="spherical", n_iter=10000000).fit(X[:year_days])
    err = 0
    try:
        hidden_statesy = model.predict(X[:year_days])
    except:
        if err <= 10:
            err += 1
            continue
        else:
            print('数据出错，修改初始数据后重新运行。')
            k += 1
            continue
    res['hidden_states'][:year_days] = hidden_statesy
    retcumsum = np.arange(model.n_components, dtype=float)

    tfy = res.closeIndex[:year_days] - res.openIndex[:year_days]
    for i in range(model.n_components):
        idxy = (hidden_statesy == i)
        idxy = np.append(0, idxy[:-1])
        res['sig_ret%s' % i][:year_days] = tfy.multiply(idxy, axis=0)
        retcumsum[i] = res['sig_ret%s' % i][:year_days].cumsum()[-1]

    seq = np.arange(model.n_components)
    for i in range(model.n_components):
        for j in range(0, i):
            if retcumsum[i] > retcumsum[j]:
                seq[i], seq[j] = seq[j], seq[i]
                retcumsum[i], retcumsum[j] = retcumsum[j], retcumsum[i]

    for i in range(model.n_components):
        idxy = (hidden_statesy == seq[i])
        idxy = np.append(0, idxy[:-1])
        res['sig_ret%s' % i][:year_days] = tfy.multiply(idxy, axis=0)

    # 制定策略
    judge = np.zeros(model.n_components)
    for i in range(model.n_components):
        longdays = 0
        shortdays = 0
        idxy = (hidden_statesy == i)
        for j in range(year_days - 1):
            if idxy[j] == 1 and tfy[j] > 0 and res['sig_ret%s' % i][:year_days][j] > 0:
                longdays += 1
            if idxy[j] == 1 and tfy[j] > 0 and res['sig_ret%s' % i][:year_days][j] > 0:
                shortdays += 1
        if longdays > 1.5 * shortdays:
            judge[i] = 1
        if shortdays > 1.5 * longdays:
            judge[i] = -1
    idxlongy = np.zeros(year_days)
    idxshorty = np.zeros(year_days)
    # judge=1做多，judge=-1做空
    for i in range(model.n_components):
        if judge[i] > 0:
            idxlongy += (hidden_statesy == seq[i])
        if judge[i] < 0:
            idxshorty += (hidden_statesy == seq[i])
    # 获得状态结果后第二天进行买入操作
    res['long_sig'][:year_days] = (idxlongy > 0)
    idxlongy = np.append(0, idxlongy[:-1])
    # 获得状态结果后第二天进行买入操作
    res['short_sig'][:year_days] = (idxshorty > 0)
    idxshorty = np.append(0, idxshorty[:-1])
    res['today_diff'][:year_days] = tfy
    res['sig_retbest'][:year_days] = tfy.multiply(idxlongy, axis=0) - tfy.multiply(idxshorty, axis=0)
    res['pre_interest'][:year_days] = res['sig_retbest'][:year_days].cumsum()

    y = 0
    err = 0
    while (y < days-year_days):
        model = GaussianHMM(n_components=16, covariance_type="spherical", n_iter=10000000).fit(X[y:y + year_days])

        try:
            hidden_statesy = model.predict(X[y:y + year_days])
            print('共%d周，' % (days - year_days), end='')
            print('这是第%d周。' % (y + 1))
        except:
            if err <= 10:
                err += 1
                continue
            else:
                res['long_sig'][y + year_days - 1] = False
                res['short_sig'][y + year_days - 1] = False
                print('共%d周，' % (days - year_days), end='')
                print('这是第%d周。' % (y + 1))
                print('出错了，下周不开仓。')
                y += 1
                err = 0
                continue
        res['hidden_states'][y:y + year_days] = hidden_statesy
        retcumsum = np.arange(model.n_components, dtype=float)

        tfy = res.closeIndex[y:y + year_days] - res.openIndex[y:y + year_days]
        for i in range(model.n_components):
            idxy = (hidden_statesy == i)
            idxy = np.append(0, idxy[:-1])
            res['sig_ret%s' % i][y:y + year_days] = tfy.multiply(idxy, axis=0)
            retcumsum[i] = res['sig_ret%s' % i][y:y + year_days].cumsum()[-1]

        seq = np.arange(model.n_components)
        for i in range(model.n_components):
            for j in range(0, i):
                if retcumsum[i] > retcumsum[j]:
                    seq[i], seq[j] = seq[j], seq[i]
                    retcumsum[i], retcumsum[j] = retcumsum[j], retcumsum[i]

        for i in range(model.n_components):
            idxy = (hidden_statesy == seq[i])
            idxy = np.append(0, idxy[:-1])
            res['sig_ret%s' % i][y:y + year_days] = tfy.multiply(idxy, axis=0)

        # 制定策略
        for i in range(model.n_components):
            longdays = 0
            shortdays = 0
            idxy = (hidden_statesy == i)
            for j in range(year_days - 1):
                if idxy[j] == 1 and tfy[j + 1] > 0 and res['sig_ret%s' % i][y:y + year_days][j] > 0:
                    longdays += 1
                if idxy[j] == 1 and tfy[j + 1] < 0 and res['sig_ret%s' % i][y:y + year_days][j] < 0:
                    shortdays += 1
            if longdays > 1.5 * shortdays:
                judge[i] = 1
            if shortdays > 1.5 * longdays:
                judge[i] = -1
        idxlongy = np.zeros(year_days)
        idxshorty = np.zeros(year_days)
        # judge=1做多，judge=-1做空
        for i in range(model.n_components):
            if judge[i] > 0:
                idxlongy += (hidden_statesy == seq[i])
            if judge[i] < 0:
                idxshorty += (hidden_statesy == seq[i])
        # 获得状态结果后第二天进行买入操作
        res['long_sig'][y + year_days - 1] = (idxlongy[-1] > 0)
        idxlongy = np.append(0, idxlongy[:-1])
        # 获得状态结果后第二天进行买入操作
        res['short_sig'][y + year_days - 1] = (idxshorty[-1] > 0)
        idxshorty = np.append(0, idxshorty[:-1])
        if res['long_sig'][y + year_days - 1] > 0:
            print('下周看涨。')
        elif res['short_sig'][y + year_days - 1] > 0:
            print('下周看跌。')
        else:
            print('下周不开仓。')
        res['today_diff'][y:y + year_days] = tfy
        res['sig_retbest'][y:y + year_days] = tfy.multiply(idxlongy, axis=0) - tfy.multiply(idxshorty, axis=0)
        res['pre_interest'][y:y + year_days] = res['sig_retbest'][y:y + year_days].cumsum()
        y += 1

    res['real_interest'] = res.closeIndex - res.openIndex

    # 本金100,000，每天以10%的仓位开仓，收盘平仓
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
        shares[i + 1] = (money[i] * store) // close[i]
        money[i + 1] = money[i]
        stop = 0
        # 看涨
        if long_sig[i] and not short_sig[i]:
            trademoney = 10 * shares[i + 1] * openi[i + 1]
            money[i + 1] = money[i] - trademoney
            if high[i + 1] >= 1.05 * openi[i + 1]:
                trademoney = 10 * shares[i + 1] * 1.05 * openi[i + 1]
            elif low[i + 1] <= 0.975 * openi[i + 1]:
                trademoney = 10 * shares[i + 1] * 0.975 * openi[i + 1]
            else:
                trademoney = 10 * shares[i + 1] * close[i + 1]
            money[i + 1] += trademoney
            money[i + 1] -= 10 * shares[i + 1]
        # 看跌
        if not long_sig[i] and short_sig[i]:
            trademoney = 10 * shares[i + 1] * openi[i + 1]
            money[i + 1] = money[i] + trademoney
            if low[i + 1] <= 0.95 * openi[i + 1]:
                trademoney = 10 * shares[i + 1] * 0.95 * openi[i + 1]
            elif high[i + 1] >= 1.025 * openi[i + 1]:
                trademoney = 10 * shares[i + 1] * 1.025 * openi[i + 1]
            else:
                trademoney = 10 * shares[i + 1] * close[i + 1]
            money[i + 1] -= trademoney
            money[i + 1] -= 10 * shares[i + 1]
        if i >= year_days - 1:
            if money[i + 1] >= money[i]:
                win += 1
            else:
                # 如果这一天的策略失败，计算这一天为止的最大回撤
                todaydrawback = (money[i] - money[i + 1]) / money[i]
                if todaydrawback > drawback:
                    drawback = todaydrawback

    res['money'] = np.array(money)
    res['shares'] = np.array(shares)

    netvalue = money[-1] / money[-daysn]
    rateofwin = win / daysn
    loses = daysn - win
    annualizedreturn = ((netvalue - 1) / DAYS) * 365
    ret = np.diff(money)
    for i in range(days - 1):
        retratio[i] = ret[i] / money[i]
    res['retratio'] = np.append(0, retratio)
    volatility_yr = np.std(retratio[-daysn + 1:], ddof=0) * np.sqrt(252)
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

    if diss > 2:
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
        print('第%s次循环，没能找到更好的模型' % (k + 1))

    k += 1

print('---------------------------------------------------------------------------------------------------------------')
stat = pd.DataFrame({'开始时间': [startday], '结束时间': [endday], '天数': [DAYS], '交易日数': [days],
                     '净值': ['%.2f' % netmax], '胜率': ['%.2f%%' % (100 * winsmax)],
                     '最大回撤': ['%.2f%%' % (100 * drawbackmin)], '年化收益率': ['%.2f%%' % (100 * annualmax)],
                     '夏普率': ['%.2f' % sharpemax]})

print('最终的结果：')
print('净值：%.2f' % netmax)
print('年化收益率：%.2f%%' % (100 * annualmax))
print('胜率：%.2f%%' % (100 * winsmax))
print('最大回撤：%.2f%%' % (100 * drawbackmin))
print('夏普率：%.2f' % sharpemax)

stat.to_excel('parameter.xls')
