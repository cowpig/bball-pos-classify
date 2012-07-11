from bs4 import BeautifulSoup
import urllib2
import re

'''
queries nba.com and returns the stats of a
given player in the following format:
[   height (in meters),
    weight (in kg)*,
    games played*,
    games started*,
    minutes per game*,
    FG%,
    3pt%,
    FT%,
    OFF,
    DEF,
    RPG,
    APG,
    SPG,
    BPG,
    TO,
    PF,
    PPG
]    *turned out to be pretty useless in logistic
    regression classifier
'''

def stats(playername):
    name = playername.replace('.','')
    name = name.replace(' ','_')
    name = name.lower()
    url = 'http://www.nba.com/playerfile/' + name + '/career_stats.html'
    html = urllib2.urlopen(url).read()
    soup = BeautifulSoup(html)
    stats1 = soup.find_all('div', {"class" : "playerInfoStatsPlayerInfoBorders"})
    table = soup.find("table", {"class" : "careerAvg"})
    table = table.find("tr", {"class" : "career"}).find_all('td')
    assert len(table) == 17
    table = table[2:16]
    non_num = re.compile('[^\\d.]*')
    
    # get height
    height = stats1[1].find('span').string.encode('ascii','ignore')
    ph = re.compile('[\\d.]+[ ]*\n')
    height = ph.findall(height)
    assert len(height) == 1
    height = float(non_num.sub('',height[0]))
    
    # get weight
    weight = stats1[2].find('span').string.encode('ascii','ignore')
    pw = re.compile('/[ ]*[\\d.]+[ ]*kg')
    weight = pw.findall(weight)
    assert len(weight) == 1
    weight = float(non_num.sub('',weight[0]))
    
    out = [height, weight]
    
    # get the rest of the stats
    for x in table:
        out.append(float(x.string.replace(',','')))
    
    assert len(out) == 16
    return out

