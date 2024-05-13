import sys
import json
import os
from discord import SyncWebhook

'''
=========
Functions
=========
'''
def send(message, channel='1'):
    return
    try:
        channel = int(os.environ.get('TWIG_CHANNEL'))
    except:
        channel = '1'
    if channel == '1':
        # no channel; revert to default
        channel = 'webhook'
    elif not channel in (['2', '3', '4', '5', '6', '7', '8', '9', '10']):
        # unknown channel; revert to default
        channel = 'webhook'
    else:
        # use given channel number
        channel = f'webhook-{channel}'

    twig_mode = os.environ.get('TWIG_MODE')
    if twig_mode is None or twig_mode == "WEB":
        websend(message, channel)
    elif twig_mode == "DROP":
        return
    elif twig_mode == "PRINT":
        print(message)
        return
        
def websend(message, channel):
    try:
        webhook = json.load(open('../twig.json', 'r'))[channel]
        webhook = SyncWebhook.from_url(webhook)
        webhook.send(message)
    except:
        pass

if __name__ == '__main__':
    message = sys.argv[1]
    send(message)
