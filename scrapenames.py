from bs4 import BeautifulSoup
import urllib2

'''
scraping the labeled data from yahoo sports
'''
def scrape(filename):
    base_url = "http://sports.yahoo.com/nba/players?type=position&c=NBA&pos="
    positions = ['G', 'F', 'C']
    players = 0

    with open(filename, 'w') as names:
        for p in positions:
            html = urllib2.urlopen(base_url + p).read()
            soup = BeautifulSoup(html)
            table = soup.find_all('table')[9]
            cells = table.find_all('td')
            
            for i in xrange(4, len(cells) - 1, 3):
                names.write(cells[i].find('a').string + '\t' + p + '\n')
                players += 1
                
    print "...success! %r players downloaded." % players
