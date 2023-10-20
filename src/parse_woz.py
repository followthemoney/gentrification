import json
import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np

load_dotenv()

PATH = os.environ.get('PATH_RAW')

with open(PATH + 'woz.json', 'r') as file:
    wozlist = []
    for line in file:
        wozdict = {}
    
        woz = json.loads(line)
        wozwaarden = woz.get('wozWaarden')
        wozobjecten = woz.get('wozObject')
        if wozobjecten is not None:
            nummeraanduiding_id = wozobjecten.get('nummeraanduidingid')
            if nummeraanduiding_id is not None:
                wozdict.update({'nummeraanduiding_id': int(nummeraanduiding_id)})
            else:
                continue
        else:
            continue

        for waarde in wozwaarden:
            if waarde.get('peildatum') == '2016-01-01':
                wozdict.update({'woz_2016': waarde.get('vastgesteldeWaarde')})
            elif waarde.get('peildatum') == '2022-01-01':
                wozdict.update({'woz_2022': waarde.get('vastgesteldeWaarde')})
        
        if wozdict.get('woz_2022') and wozdict.get('woz_2016') is not None:
            wozlist.append(wozdict)

df = pd.DataFrame(wozlist)
del wozlist
df.to_csv(PATH + 'wozobjecten_2016_2022.csv', index=False)

