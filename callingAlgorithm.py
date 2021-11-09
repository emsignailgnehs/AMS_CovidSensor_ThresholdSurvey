from app.utils.picoLibrary import Pico
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from scipy import signal
# import traceback
from scipy.optimize import least_squares

RESULT_CODE = {1:'Positive',0:'Negative',-1:'Perror'}


def CoivdResultAnalysis(data, mode=None, reason=None):
    """
    This is the entry point called by app to give results.
    All measurement data are given in 'data'.
    mode is 'end' or 'abort' or 'rePredict'.
    if mode is abort, reason is given.
    This method will return status and result.
    Status is a dictionary, result is a string.
    result: a string indicating Positive or Negative or Invalid, or Error.
    status: {
        'stat': 'ok',
        'avgTemp': average temperature during measurement.
        'chipInsertion': n/m, (n out of m measurement that chip is inserted.)
        'fluid': similar to chipInsertion, but with each channel.
    }
    """
    if mode == 'abort':
        status = {'stat': 'abort', 'reason': reason}
        return status, 'Error'
    if mode == 'rePredict':
        return None, resultToString(predictAllChannelFromScan(data.get('scan', {}),curveQC=False))

    temp = data.get('temperature', {}).get('data', [0])
    scan = data.get('scan', {})
    chip = data.get('chipInsertion', {}).get('data', [])
    fluid = data.get('fluidFill', {})

    # the rules to make the call.
    stat = ''
    fluidResult = []
    fluidFillError = 0
    for c, cd in fluid.items():
        filled = sum(
            i >= Pico.FLUID_FILL_THREASHOLD for i in cd.get('data', []))
        total = len(cd.get('data', []))
        fluidResult.append(f"{c}-{filled}/{total}")
        fluidFillError = max((total - filled)/total, fluidFillError)

    if fluidFillError >= 0.25:
        stat = 'F'

    avgT = (sum(temp) / len(temp)) if len(temp) else 0
    # if temperature error is over 5deg, report error
    if abs(avgT - 65) > 5:
        stat += 'T'
    avgTemp = f"{avgT:.2f} C"

    chipInsertRatio = (sum(chip) / len(chip)) if len(chip) else 0
    # if more than 1/4 the time chip was not inserted, stat is error
    if chipInsertRatio <= 0.75:
        stat += 'C'
    chipInsert = f"{sum(chip)} / {len(chip)}"

    status = {
        'stat': 'ok', # if there is error, will be F,T,C for fluid fill error, temperature error, chip insertion error.
        'avgTemp': avgTemp,
        'chipInsertion': chipInsert,
        'fluid': ', '.join(fluidResult),
    }

    # if the stat is not 'ok', then return inavlid result.
    # 20210513 Hui always predict the result even if the fluid fill/chip insertion/ etc have error.
    # because I have the curve QC in place, if the curve is so obviously wrong, it will fail.
    # and I notice sometimes the current fluid fill detection will fire wrongly occasionaly.
    if stat:
        status['stat'] = stat
        # return status, f'Invalid [{stat}]'
    # else:
    channelResult = predictAllChannelFromScan(scan)
    # handle the logic of caling Positive or negative based on channel result.
    result = ResultInterpretation(channelResult)
    status['channelResult'] = resultToString(channelResult)
    return status, result


def ResultInterpretation(r):
    """
    The r is a dictionary, returned from predictAllChannelFromScan.

    this function return, Positive, Negative, Invalid [R | E]
    if invalid, the letter in baracket is used to indicate resason for invalid.
    eitehr R: RNaseP is negative, or E: prediction error
    20210428: for capcat, we loaded C1 as N7, C4 as RP4
    20210520: currently both channel will be N7.
    20210
    """
    N,Nct,Npr,Nsd = r.get('C1',[-1,30,0,0])
    R,Rct,Rpr,Rsd = r.get('C4',[-1,30,0,0])
    if N!=R and N+R == 1:
        # if there is 1 positive 1 negative,
        # and if the positive if very positive, then return positive
        # if the positive is on the edge, return negative.
        ct = N* Nct + R * Rct
        pr = N* Npr + R * Rpr
        sd = N* Nsd + R * Rsd
        # 20210707 lower this SD threshold to 0.13, because recent DSM data
        # suggest signal drop for positives are not as obvious.
        if ct<=18 and pr>=0.5 and sd > 0.13:
            return 'Positive'
        else:
            return 'Negative'

    return {
        (1,1):'Positive',
        (0,0):'Negative',
        (-1,-1):'Error',
        (1,0):'Positive',
        (1,-1):'Positive',
        (0,-1):'Negative',
        (0,1):'Positive',
        (-1,0):'Negative',
        (-1,1):'Positive'
    }.get((N,R),'Invalid')


def predictAllChannelFromScan(scan,curveQC=True):
    """
    data.scan
    scan: {
      C1:   channnelData // see PositiveNegativeCall to see channelData format.
    }
    return prediciton result for each channel as a dictionary.
    {
        C1: [-1,0]: first digit is predictin, -1 means or, 0 means negative 1 means positive.
        following digits are ct,prominence,score
    }
    """
    result = {}
    for channel, channelData in scan.items():
        try:
            p_n,ct,prominence,sd= PositiveNegativeCall(channelData,channel,curveQC)
        except Exception as e:
            p_n,ct,prominence,sd = (-1,30,0,0)
        result[channel] = [p_n,ct,prominence,sd]
    return result

def resultToString(result):
    "turn the result dictionary from predictAllChannelFromScan to a string"
    return ','.join( f"{k}-{RESULT_CODE.get(v[0],'weird')} Ct:{v[1]:.1f} Pr:{v[2]:.1f} Sd:{v[3]:.2f}" for k,v in result.items())


def removeOutlier(an_array):
    "remove outlier from an list. by 3 std away"
    if len(an_array) == 0:
        return np.empty(0)
    an_array = np.array(an_array)
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    max_deviations = 3
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return an_array[not_outlier]


def curveQC(fitres):
    """
    d is a list of fitting results
    QC a current curve, before predicting if it's positive or negative.
    Return True if pass, return False if failed.
    Currently, calculating the average current
    and the CV of neighbor datapoint delta (smoothness).
    if the average and smoothness fall away from certain threshold, then give invalid result.
    """
    d = [i['pc'] for i in fitres]
    d = removeOutlier(d)
    if len(d) < 50:
        # if removed outlier there is less than 50 data points, then
        # this is invalid.
        return False

    delta = np.array(d[0:-1])-np.array(d[1:])
    avg = abs(np.array(d).mean())
    cv = np.std(delta) / (avg + 1e-6)

    if avg > 100 or avg<2:
        # if average current is above 100uA or lower than 2uA
        return False
    if cv > 0.35:
        # if cv is larger than 35%
        return False
    if (sum([i['err'] for i in fitres]) / len(fitres)) > 0.15:
        # if the average fitting error is larger than a threashold: then raise a prediction error.
        # this threshold is dertermined from data around 20210604 - 20210701.
        # the max average error seen in this period is ~0.12. Majority averageError < 0.06.
        return False
    return True



def PositiveNegativeCall(channelData,channel,qc=True):
    """
    channel data is
    {
        time: [],
        rawdata:[[v,a]...],
        fit:[{fx,fy,pv,pc,err}...]
    }
    channel is C1, C2
    return 1 as positive, 0 as negative and a decision_function score.
    5/26: added ct value caller and use that as result calling also return ct and prominence
    """
    t = channelData['time']
    if qc and not curveQC(channelData['fit']):
        # if curve QC failed, return QCerror, indicated by Ct=0, prominence and sd =0
        return -1,0,0,0
    pc = [i['pc'] for i in channelData['fit']]

    X = convert_list_to_X([[t, pc]])
    # classifier = {'C4':RPclassifier}.get(channel,Nclassifier)
    # res = classifier.predict(X)
    # score = classifier.decision_function(X)[0]
    caller = {'C1':NCT_Caller,'C4':RPCT_Caller}.get(channel,NCT_Caller)

    ctP,ct,prominence,sd = caller.transform(X)[0]

    return int(ctP),ct,prominence,sd
    # if res[0] == 1:
    #     return 'positive',score
    # else:
    #     return 'negative',score
    # return 'negative'
    # return np.random.choice(['positive','negative'])


def convert_list_to_X(data):
    """
    data is the format of:
    [[ [t1,t2...],[c1,c2...]],...]
    convert to numpy arry, retain the list of t1,t2... and c1,c2...
    """
    if not data:
        return np.array([])
    X = np.empty((len(data), 2), dtype=list)
    X[:] = data
    return X



def smooth(x, windowlenth=11, window='hanning'):
    "windowlenth need to be an odd number"
    s = np.r_[x[windowlenth-1:0:-1], x, x[-2:-windowlenth-1:-1]]
    w = getattr(np, window)(windowlenth)
    return np.convolve(w/w.sum(), s, mode='valid')[windowlenth//2:-(windowlenth//2)]


def timeseries_to_axis(timeseries):
    "convert datetime series to time series in minutes"
    return [(d-timeseries[0]).seconds/60 for d in timeseries]




def normalize(row):
    return row/np.max(row)


def reject_outliers(time, data, stddev=2):
    '''remove the outlier from time series data,
    stddev is the number of stddev for rejection.
    '''
    sli = abs(data - np.mean(data)) < stddev * np.std(data)
    return np.array(time)[sli], np.array(data)[sli]


def extract_timepionts(time, data, cutoffStart, cutoffEnd=60, n=150):
    '''
    extract time and data with time<=cutoff
    n points of data will be returned.
    unknown values are from interpolation.
    '''
    tp = 0
    datalength = len(data)
    newdata = []
    endslope = np.polyfit(time[-11:], data[-11:], deg=1)
    for i in np.linspace(cutoffStart, cutoffEnd, n):
        for t in time[tp:]:
            if t >= i:
                break
            tp += 1
        if tp+1 >= datalength:
            # if the new timepoint is outside of known range, use last 11 points to fit the curve.
            newdata.append(i*endslope[0]+endslope[1])
        else:
            # otherwise interpolate the data.
            x1 = time[tp]
            y1 = data[tp]
            x2 = time[tp+1]
            y2 = data[tp+1]
            newdata.append(y1 + (i-x1)*(y2-y1)/(x2-x1))
    return np.array(newdata)


def findTimeVal(t,val,t0,dt):
    """
    t:   [.............]
    val: [.............]
    t0:       |      ; if t0 is less than 0, then start from 0
    dt:       |---|  ; must > 0
    return:  [.....]
    find the fragment of time series data,
    based on starting time t0 and time length to extract
    assuming t is an evenly spaced time series
    """
    t0idx = int((t0 - t[0]) / (t[-1]-t[0]) * len(val))
    t1idx = int((t0 +dt - t[0]) / (t[-1]-t[0]) * len(val))
    return val[max(0,t0idx):t1idx]




class Smoother(BaseEstimator,TransformerMixin):
    def __init__(self,stddev=2,windowlength=11,window='hanning'):
        self.stddev = stddev
        self.windowlength = windowlength
        self.window = window
    def fit(self,X,y=None):
        return self
    def transformer(self,X):
        t,pc = X
        t,pc = reject_outliers(t,pc,stddev=self.stddev)
        pc = smooth(pc,windowlenth=self.windowlength,window=self.window)
        return [t,pc]
    def transform(self,X,y=None):
        # return np.apply_along_axis(self.transformer,1,X,)
        return np.array([self.transformer(i) for i in X],dtype='object')


class Derivitive(BaseEstimator,TransformerMixin):
    def __init__(self,window=31,deg=3):
        self.window = window
        self.deg = deg


    def fit(self,X,y=None):
        return self
    def transformer(self,X):
        t,pc = X
        ss = savgol_filter(pc,window_length=self.window,polyorder=self.deg,deriv=1)
        return [t,-ss,pc]
    def transform(self,X,y=None):
        # return np.apply_along_axis(self.transformer,1,X,)
        return np.array([self.transformer(i) for i in X],dtype='object')


class FindPeak(BaseEstimator,TransformerMixin):
    def __init__(self,heightlimit=0.9,widthlimit=0.05):
        self.heightlimit = heightlimit
        self.widthlimit = widthlimit

    def fit(self,X,y=None):
        return self
    def transformer(self,X):

        t,gradient,pc = X
        heightlimit = np.quantile(np.absolute(gradient[0:-1] - gradient[1:]), self.heightlimit)
        peaks,props = signal.find_peaks(gradient,prominence=heightlimit,width= len(gradient) * self.widthlimit, rel_height=0.5)


        peak_pos,left_ips,peak_prominence,peak_width = (t[-1],t[-1],0,0)
        sdAtRightIps,sdAt3min,sdAt5min,sdAt10min,sdAt15min,sdAtEnd = (0,0,0,0,0,0)
        if len(peaks) != 0:
        # most prominent peak in props
            tspan = t[-1]-t[0]
            normalizer =  tspan / len(gradient)
            maxpeak_index = props['prominences'].argmax()
            peak_pos = peaks[maxpeak_index] * normalizer + t[0]
            peak_prominence = props['prominences'][maxpeak_index]
            peak_width = props['widths'][maxpeak_index] * normalizer
            left_ips = props['left_ips'][maxpeak_index] * normalizer  + t[0]

            pcMaxIdx = len(pc) - 1

            # siganl at left_ips:
            startPosition = int(props['left_ips'][maxpeak_index])
            sStart = pc[startPosition]
            # find signal drop at different positions:
            # sigal drop at peak_width
            sdAtRightIps = sStart - pc[min(int(props['right_ips'][maxpeak_index]), pcMaxIdx)]
            # signal drop at 3 min later
            sdAt3min = sStart - pc[min(startPosition + int(3 / normalizer), pcMaxIdx)]
            # signal drop at 5 min later
            sdAt5min = sStart - pc[min(startPosition + int(5 / normalizer), pcMaxIdx)]
            # signal drop at 10 min later
            sdAt10min = sStart - pc[min(startPosition + int(10 / normalizer), pcMaxIdx)]
            # siganl drop at 15 min later
            sdAt15min = sStart - pc[min(startPosition + int(15 / normalizer), pcMaxIdx)]
            # signal drop at end
            sdAtEnd = sStart - pc[-1]
        return [left_ips,peak_prominence*100,peak_width,sdAtRightIps,sdAt3min,sdAt5min,sdAt10min,sdAt15min,sdAtEnd,t,gradient,pc]

    def transform(self,X,y=None):
        # return np.apply_along_axis(self.transformer,1,X,)
        return np.array([self.transformer(i) for i in X],dtype='object')



class HyperCt(BaseEstimator,TransformerMixin):
    "calculate the Ct from threshold method,the threshold line is from a hyperbolic fitting"
    def __init__(self,offset=0.05):
        """
        offset is how much the fitted curve shifts down. this is in relative scale to the intial fitting point.
        """
        self.offset = offset

    def fit(self,X,y=None):
        return self

    def hyper(self,p,x,y):
        return p[0]/(x+p[1]) +p[2] -y
    def hyperF(self,p):
        return lambda x:p[0]/(x+p[1]) +p[2]

    def transformer(self,X):
        offset = self.offset
        t,deri,smoothed_c = X[-3:]
        left_ips,peak_prominence,peak_width = X[0:3]
        tofit = findTimeVal(t,smoothed_c,t[0],left_ips - t[0])

        fitres = least_squares(self.hyper,x0=[5,5,0.5],
                    args=(np.linspace(t[0],left_ips,len(tofit)),tofit))
        fitpara = fitres.x

        thresholdpara = fitpara - np.array([0,0,(tofit[-1]) * offset])
        thresholdline = self.hyperF(thresholdpara)
        tosearch = findTimeVal(t,smoothed_c,left_ips,t[-1])
        tosearchT = np.linspace(left_ips,t[-1],len(tosearch))
        thresholdSearch = thresholdline(tosearchT) - tosearch
        thresholdCt = left_ips
        for sT,sthre in zip(tosearchT,thresholdSearch):
            if sthre > 0:
                break
            thresholdCt = sT
        return  [*X[0:-3],thresholdCt]

    def transform(self,X,y=None):
        return np.array([self.transformer(i) for i in X])


class Normalize(BaseEstimator,TransformerMixin):
    """
    Transformer to normalize an array with given parameters
    params:
    mode: str, can be 'max', 'mean',
    dataTimeRange: float, describe the total length of data in minutes.
    normalzieToTrange: (), tuple, describe from and to time in minutes it will normalize to.

    """
    def __init__(self,mode='max',normalizeRange=(5,20)):
        self.mode=mode
        self.normalizeRange = normalizeRange
        self.q_ = {'max':np.max,'mean':np.mean}.get(self.mode,None)
        self.from_ = self.normalizeRange [0]
        self.to_ = self.normalizeRange [1]

    def fit(self,X,y=None):
        return self

    def transformer(self,X):

        time,pc = X
        f = np.abs(np.array(time) - self.from_).argmin()
        t = np.abs(np.array(time) - self.to_).argmin()
        normalizer = max(self.q_(pc[f:t]), 1e-3)
        return time,pc/normalizer

    def transform(self,X,y=None):
        return np.array([self.transformer(i) for i in X],dtype='object')


class Truncate(BaseEstimator,TransformerMixin):
    """
    Transformer to Truncate and interpolate data,
    input X is a time and current 2d array.
    [0,0.3,0.6...] in minutes,
    [10,11,12...] current in uA.
    return a 1d data array, with n data points, start from cutoffStart time,
    end at cutoffEnd time. Time are all in minutes.
    """
    def __init__(self,cutoffStart,cutoffEnd,n):
        self.cutoffStart = cutoffStart
        self.cutoffEnd = cutoffEnd
        self.n = n
    def fit(self,X,y=None):
        return self
    def transformer(self,X,y=None):
        t,pc = X
        c = (self.cutoffEnd - self.cutoffStart) / (t[-1] - t[0]) * len(t)
        # i have to do this float conversion, otherwise I got dtype inexact problem in polyfit.
        newcurrent = extract_timepionts(np.array([float(i) for i in t]),
                                        np.array([float(i) for i in pc]),self.cutoffStart,self.cutoffEnd,self.n)
        return np.linspace(self.cutoffStart,self.cutoffEnd,int(c)),newcurrent
    def transform(self,X,y=None):
        return np.array([self.transformer(i) for i in X],dtype='object')


class CtPredictor(BaseEstimator,TransformerMixin):
    "a predictor to predict result based on ct and prominence threshold from FindPeak"
    def __init__(self,ct=25,prominence=0.2,sd=0.1):
        self.ct=ct
        self.prominence = prominence
        self.sd=sd

    def fit(self,X,y=None):
        return self

    def transformer(self,x):
        "return 0,1 flag, ct, prominence, signal drop at 5min"
        return int(x[-1]<=self.ct and x[1]>=self.prominence and x[5]>=self.sd),x[-1],x[1],x[5]

    def transform(self,X,y=None):
        return np.apply_along_axis(self.transformer,1,X)



NCT_Caller = Pipeline([
    ('smooth',Smoother(stddev=2,windowlength=11,window='hanning')),
    ('normalize', Normalize(mode='mean',normalizeRange=(5,10))),
    ('truncate',Truncate(cutoffStart=5,cutoffEnd=30,n=90)),
    ('Derivitive',Derivitive(window=31,deg=3)),
    ('peak',FindPeak()),
    ('hyperCt',HyperCt()),
    ('predictor',CtPredictor(ct=20.4,prominence=0.4,sd=0.131))
])

RPCT_Caller = Pipeline([
    ('smooth',Smoother(stddev=2,windowlength=11,window='hanning')),
    ('normalize', Normalize(mode='mean',normalizeRange=(5,10))),
    ('truncate',Truncate(cutoffStart=5,cutoffEnd=30,n=90)),
    ('Derivitive',Derivitive(window=31,deg=3)),
    ('peak',FindPeak()),
    ('hyperCt',HyperCt()),
    ('predictor',CtPredictor(ct=20.4,prominence=0.4,sd=0.131))
])
