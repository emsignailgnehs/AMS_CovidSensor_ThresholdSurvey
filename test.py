"""
imports
-------------------------------------------------------------------------------------------------------"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import textwrap
import csv
from itertools import combinations
from tkinter import Tk, filedialog
from compress_pickle import dump,loads
from pathlib import Path
import .callingAlgorithm

"""
functions
-------------------------------------------------------------------------------------------------------"""
def ui_getfilenames(_filetype = ''):
    """
    _filetype is "a list" of "tuples":

    [
    ('filename1', '*.extension'),
    ('filename2', '*.extension'), ...
    ]

    """
    root = Tk() # pointing root to Tk() to use it as Tk() in program.
    root.withdraw() # Hides small tkinter window.
    root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.

    if _filetype == '':
        filez = filedialog.askopenfilenames(parent=root, title='Choose a file/ files')
    else:
        filez = filedialog.askopenfilenames(parent=root, title='Choose a file/ files', filetypes = _filetype)

    # return [repr(filename) for filename in filenames]
    return list(root.tk.splitlist(filez))

# def removeDuplicates(*args):
    currents = set()
    ids = []
    for t,c in args[0]:
        if sum(c) in currents:
            ids.append(False)
        else:
            ids.append(True)
            currents.add(sum(c))
    return [i[ids] for i in args]

# def timeseries_to_axis(timeseries):
    "convert datetime series to time series in minutes"
    return [(d-timeseries[0]).seconds/60 for d in timeseries]

# def convert_list_to_X(data):
    """
    data is the format of:
    [[ [t1,t2...],[c1,c2...]],...]
    convert to numpy arry, retain the list of t1,t2... and c1,c2...
    """
    if not data: return np.array([])
    X = np.empty((len(data),2),dtype=list)
    X[:] = data
    return X

# def findTimeVal(t,val,t0,dt):
    t0idx = int((t0 - t[0]) / (t[-1]-t[0]) * len(val))
    t1idx = int((t0 +dt - t[0]) / (t[-1]-t[0]) * len(val))
    return val[max(0,t0idx):t1idx]

"""
classes

Note:
1. the ViewerDataSource here is truncated for legibility
-------------------------------------------------------------------------------------------------------"""
class ViewerDataSource():
    def __init__(self,app=None):
        """
        self.pickles: {'file': {'data': pickledata file, 'modified': True/False}}
        self.dateView: {datetime in days: [ orderred experiment data {name: exp: ...}], 'deleted': []}
        """
        self.pickles = {}
        self.dateView = {'deleted':[]}
        self.expView = {'deleted':[]}
        self.rawView = {'deleted':[],}
        self.viewsType = (self.expView,self.dateView,self.rawView)
        self.picklefolder = ""
        self.app = app

    def print(self,msg):
        if self.app:
            self.app.displayMsg(msg)
        else:
            print(msg)

    @property
    def needToSave(self):
        for f,d in self.pickles.items():
            if d['modified']:
                return True
        return False

    @property
    def needToSaveToMonitor(self):
        return ( self.pickles.get('monitorMemory',{}).get('modified',False)
        or self.pickles.get('picoMemory',{}).get('modified',False) )

    def save(self,callback=None):
        for f,d in self.pickles.items():
            if d['modified']:
                if f in ('monitorMemory' , 'picoMemory'):
                    continue
                else:
                    if f.endswith('.pickle'):
                        tosave = f+'z'
                    elif f.endswith('.gz'):
                        tosave = f[0:-2] + '_converted.picklez'
                    elif f.endswith('unspecified_filename_in_load_reader_data'):
                        tosave = d.get('tempSavePath','./readerDownloadedData.picklez')
                    else:
                        tosave = f
                    with open(tosave,'wb') as o:
                        dump(d['data'],o,compression='gzip')
                    d['modified'] = False
                self.print(f'Saved <{tosave}>.')
        if callback:callback()

    def memorySave(self):
        memorySave = {}
        for f,d in self.pickles.items():
            if d['modified']:
                if f in ('monitorMemory' , 'picoMemory'):
                    memorySave[f] = d['data']['pstraces']
                    d['modified'] = False
        return memorySave

    def remove_all(self):
        'remvoe all data'
        self.pickles = {}
        self.dateView = {'deleted':[]}
        self.expView = {'deleted':[]}
        self.rawView = {'deleted':[]}
        self.picklefolder = ""

    def load_device_data(self,data):
        """
        load data from device and convert format.
        """
        pstrace = {}
        for packet in data:
            try:
                meta = packet['meta']
                dateString = meta.get('created',None) or packet.get('createdAt',None)

                if not dateString: # set a default date if no date is created.
                    dateString = '1111-01-01T13:24:57.817016'

                created = datetime.fromisoformat(dateString)
                channel = meta.get('device','?Device')
                _id = packet['_id']

                _scandata = packet.get('data',{})
                if not _scandata: continue
                scan = _scandata.get('scan',None)
                if scan:
                    for chipChannel, channelData in scan.items():

                        t = [created + timedelta(minutes=i) for i in channelData['time'] ]
                        if t:
                            raw = {
                                'time': t,
                                'rawdata': [ [np.linspace(*v).tolist(),a]  for v,a,*_ in channelData['rawdata']],
                                'fit': channelData['fit']
                            }
                            psTraceChannel = dict(
                                    _id = _id,
                                    name = meta.get('name','No Name')+'-'+chipChannel,
                                    exp = meta.get('exp','No Exp'),
                                    dtype='device-transformed')

                            psTraceChannel.update(data=raw)

                            desc = f"{meta.get('desc','No Desc')} | "
                            desc += f"{json.dumps(meta)} | "
                            desc += f"{json.dumps(packet['status'])} | "
                            desc += f"{json.dumps(packet['result'])}"
                            psTraceChannel.update(desc=desc)

                            if channel in pstrace:
                                pstrace[channel].append(psTraceChannel)
                            else:
                                pstrace[channel] = [psTraceChannel]
            except Exception as e:
                self.print(f"ViewerDataSource.load_device_data error: {e}")
                continue
        return {'pstraces': pstrace}

    def load_reader_data(self,addrs):
        "load data from reader."
        # check which pickle file is the device data.
        deviceIdx = {}
        readerDatacount = 0
        self.readerFile = None
        for file,data in self.pickles.items():
            if data['data'].get('isReaderData',None):
                self.readerFile = file
                readerDatacount += 1
                for deviceId,deviceData in data['data']['pstraces'].items():
                    if deviceData:
                        for _d in deviceData[::-1]:

                            if not _d.get('deleted',False):
                                lastid = _d['_id']
                                print('Download after : ', _d['name'])
                                deviceIdx[deviceId] = lastid
                                break
        # if more than one readerData, then abort reading because the last id will be confused.
        if readerDatacount > 1:
            self.print('ViewerDataSource.load_reader_data: More than one reader data loaded. Return.')
            return

        # if no reader data is loaded, create new with default file location.
        if not self.readerFile:
            self.readerFile = './unspecified_filename_in_load_reader_data'
            self.pickles[self.readerFile] = {'data':{'pstraces':{},'isReaderData':True},'modified':True}

        self.pickles[self.readerFile]['modified'] = True # set modified to true.

        for deviceAddr in addrs:
            ws = WSClient(deviceAddr,self)
            if ws.con:
                idx = deviceIdx.get(deviceAddr,None)
                if idx:
                    data = ws.send('dataStore.getDataAfterIndex',index=idx,pwd="",returnRaw=True)
                    data = json.loads(data)
                    items = data.get('data',[])
                else:
                    items = []
                    perPage = 5
                    page=0
                    while True:
                        self.print(f'{deviceAddr} - Getting Data {page*perPage} - {(page+1)*perPage}')
                        data = ws.send('dataStore.getRecentPaginated',page=page,perpage=perPage,pwd="",returnRaw=True)
                        data = json.loads(data)
                        newItems = data.get('data',{}).get('items',[])
                        items.extend(newItems)
                        # print(data.keys(),len(newItems))
                        page+=1
                        if len(newItems) < perPage:
                            break
                ws.close()
                items = items[::-1] # reverse the order because the new data are in descending order of date.
                if items:
                    firstId = items[0]['_id'] #get _id of the first data.
                    loadedpstraces = self.load_device_data(items)['pstraces']
                    for deviceAddr,deviceData in loadedpstraces.items():
                        self.print(f"Received <{len(deviceData)}> data from {deviceAddr}.")
                        #merge new data with old.
                        pst = self.pickles[self.readerFile]['data']['pstraces']
                        toRemove = 0
                        for d in pst.get(deviceAddr,[])[::-1]:
                            if d.get('_id',None) == firstId:
                                toRemove +=1
                            else:
                                break
                        pst[deviceAddr] = pst.get(deviceAddr,[])[:-toRemove] + deviceData
        self.rebuildViews()

    def load_picklefiles(self,files):
        for file in files:
            compression = 'gzip' if file.endswith('.picklez') else None
            with open(file, 'rb') as f:
                data = f.read()
            try:
                data = loads(data,compression=compression)
            except ValueError as e:
                print('load_picklefiles with pickle 5',e)
                dec = gzip.decompress(data)
                data = pickle5.loads(dec)

            if file.endswith('.gz'):
                try:
                    data = self.load_device_data(data)
                    file = file.rsplit('.')[0]+'.picklez'
                except Exception as e:
                    self.print(f'load_picklefiles error: {e}')
                    continue
            self.pickles[file] = {'data':data,'modified':False}
            self.picklefolder = Path(file).parent
        self.rebuildViews()

    def load_from_memory(self,data):
        self.pickles[data['source']] = {'data': {'pstraces':data['pstraces']}, 'modified':False}
        self.rebuildViews()

    def modify(self,d,key,value):
        d[key]=value
        self.pickles[d['_file']]['modified'] = True

    def rebuildViews(self):
        self.rebuildDateView()
        self.rebuildExpView()
        self.rebuildRawView()
        self.viewsType = (self.expView,self.dateView,self.rawView)

    def rebuildDateView(self):
        ""
        self.dateView = {'deleted':[]}
        for file,data in self.pickles.items():
            dataset = data['data']['pstraces']
            for channel, cdata in dataset.items(): # cdata is the list of chanel data
                for edata in cdata: # edata is each dictionary of a timeseries tracing.
                    date = edata['data']['time'][0].replace(hour=0,minute=0,second=0,microsecond=0,tzinfo=None)
                    deleted = edata.get('deleted',False)
                    edata['_file'] = file
                    edata['_channel'] = channel
                    # update new data to folder view
                    if deleted:
                        self.dateView['deleted'].append(edata)
                        continue
                    if date in self.dateView:
                        self.dateView[date].append(edata)
                    else:
                        self.dateView[date] = [edata]
        # sort new views by date
        for k,item in self.dateView.items():
            item.sort(key = lambda x: (x['name'],x['data']['time'][0]))

    def rebuildExpView(self):
        ""
        self.expView = {'deleted':[]}
        for file,data in self.pickles.items():
            dataset = data['data']['pstraces']
            for channel, cdata in dataset.items(): # cdata is the list of chanel data
                for edata in cdata: # edata is each dictionary of a timeseries tracing.
                    exp = edata['exp'] if edata['exp'] else 'Unassigned'
                    deleted = edata.get('deleted',False)
                    edata['_file'] = file
                    edata['_channel'] = channel
                    if deleted:
                        self.expView['deleted'].append(edata)
                        continue

                    if exp in self.expView:
                        self.expView[exp].append(edata)
                    else:
                        self.expView[exp] = [edata]
        # sort new views by date.
        for k,item in self.expView.items():
            item.sort(key = lambda x: (x['name'],x['data']['time'][0]))

    def rebuildRawView(self):
        self.rawView = {'deleted':[],'data':[]}
        for file,data in self.pickles.items():
            dataset = data['data']['pstraces']
            for channel, cdata in dataset.items(): # cdata is the list of chanel data
                for edata in cdata: # edata is each dictionary of a timeseries tracing.
                    deleted = edata.get('deleted',False)
                    edata['_file'] = file
                    edata['_channel'] = channel
                    if deleted:
                        self.rawView['deleted'].append(edata)
                        continue
                    self.rawView['data'].append(edata)
        # sort new views by date.
        for k,item in self.rawView.items():
            item.sort(key = lambda x: (x['name'],x['data']['time'][0]))

    def sortViewByNameOrTime(self,mode='time'):
        "sort items in views by their name or time."
        if mode == 'time':
            sortkey = lambda x: (x['data']['time'][0],x['name'])
        elif mode == 'name':
            sortkey = lambda x: (x['name'],x['data']['time'][0])
        elif mode == 'result':
            sortkey = lambda x: {None:3,'positive': 0, 'negative': 1}[x.get('userMarkedAs',None)]
        elif mode == 'predict':
            sortkey = lambda x: {None:3,'positive': 0, 'negative': 1,'failed': 2}[x.get('predictAs',None)]
        for view in self.viewsType:
            for k, item in view.items():
                item.sort(key= sortkey)

    def itemDisplayName(self,item):
        result ={None:'','positive': "✅", 'negative':'❌'}[item.get('userMarkedAs',None)]
        call ={None:'','positive': "✅", 'negative': '❌','failed':'❓'}[item.get('predictAs',None)]
        if call:
            tag = (result or '⛔️') + call
        else:
            tag = result
        return tag+item['_channel']+'-'+item['name']

    def generate_treeview_menu(self,view='dateView'):
        "generate orderred data from self.pickles"
        Dataview = getattr(self,view)
        keys = list(Dataview.keys())
        keys.remove('deleted')
        if view == 'dateView':
            keys.sort(reverse=True)
            keys = [(k.strftime('%Y / %m / %d'), [ ( f"{k.strftime('%Y / %m / %d')}$%&$%&{idx}" ,
            self.itemDisplayName(item) ) for idx,item in enumerate(Dataview[k]) ]) for k in keys]
        elif view in ('expView','rawView'):
            keys.sort()
            keys = [(k , [(f"{k}$%&$%&{idx}", self.itemDisplayName(item) )
                    for idx,item in enumerate(Dataview[k])] ) for k in keys]

        keys.append(('deleted', [ (f"deleted$%&$%&{idx}" ,self.itemDisplayName(item))
            for idx,item in enumerate(Dataview['deleted'])] ))

        return keys

    def getData(self,identifier,view,):
        "get data from view with identifier"
        res = identifier.split('$%&$%&')
        if len(res) != 2:
            return None
        key,idx = res
        if view=='dateView':
            key = datetime.strptime(key ,'%Y / %m / %d') if key!='deleted' else key
        return getattr(self,view)[key][int(idx)]

    def exportXy(self):
        """export all data in date source that have userMarkedAs
        export X and y;
        X is the format of numpy array, [[list, list]...]
        """
        data = self.rawView.get('data',[])
        results = []
        for d in data:
            if d.get('userMarkedAs',None):
                t = timeseries_to_axis(d['data']['time'])
                pc = [i['pc'] for i in d['data']['fit']]
                results.append([(t,pc),int(d['userMarkedAs']=='positive'), d.get('name','No Name'),d.get('_channel','Unknown')])
        results.sort(key=lambda x:(x[1],x[2]))
        traces = [i[0] for i in results]
        userMark = [i[1] for i in results]
        names = [i[2] for i in results]
        devices = [i[3] for i in results]
        return convert_list_to_X(traces),np.array(userMark),np.array(names),np.array(devices)

    def predict(self,clf,callback=print):
        "run prediction with clf on its data. run callback for each run."
        positive = 0
        negative = 0
        failed = 0
        for d in self.rawView.get('data',[]):
            t = timeseries_to_axis(d['data']['time'])
            pc = [i['pc'] for i in d['data']['fit']]
            try:
                res = clf.predict(convert_list_to_X([(t,pc),] ))
                if res[0] == 1:
                    positive += 1
                    d['predictAs'] = 'positive'
                elif res[0] == 0:
                    negative += 1
                    d['predictAs'] = 'negative'
            except Exception as e:
                failed += 1
                d['predictAs'] = 'failed'
                callback(f'{d["name"]} - prediction error: {e}')
        callback(f"Total: {positive+negative+failed}, Positive: {positive}, Negative: {negative}, Failed: {failed}")
        for k in self.pickles.values():
            k['modified'] = True

"""
main/tests
-------------------------------------------------------------------------------------------------------"""
"""
LOAD FILES
"""
dataSource = ViewerDataSource()

# pickleFiles = [picklefile]
pickleFiles = ui_getfilenames([('Pickle files', '*.picklez')])

dataSource.load_picklefiles(pickleFiles)
X, y, names,devices = removeDuplicates(*dataSource.exportXy())

"""

"""

"""
tests
-------------------------------------------------------------------------------------------------------"""
pickleFiles
type(X)
plt.plot(X[0][0], X[0][1])

""
Goal:
1. mine Ct, SD, Prominence
    for each dataset
2. visualize the distribution
3. automatically map outliers
"""
